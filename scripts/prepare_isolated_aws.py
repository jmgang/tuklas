"""Prepare both distinct hosts and launch the controller only after readiness.

This runs locally with admin-general; inference and timings run on AWS only.
"""
import json
from pathlib import Path
import shlex
import sys
import time
from uuid import uuid4
import boto3
from botocore.config import Config


def main(directory):
    directory=Path(directory); r=json.loads((directory / "resources.json").read_text())
    s=boto3.Session(profile_name="admin-general",region_name="ap-southeast-1")
    c=s.client("ssm",config=Config(connect_timeout=5,read_timeout=30,retries={"max_attempts":0}))
    commands={}
    for role in ("client","server"):
        settings={**r,"local_resource_id":r[role+"_instance"]}
        body=f"""import boto3,hashlib,json,pathlib,socket,tarfile
settings=json.loads({json.dumps(settings)!r})
base=pathlib.Path('/opt/pycon-isolated'); raw=base/'input'; code=base/'code'
raw.mkdir(exist_ok=False);code.mkdir(exist_ok=False)
s=boto3.client('s3',region_name='ap-southeast-1')
def unpack(bucket,key,target,expected=None):
 response=s.get_object(Bucket=bucket,Key=key)
 expected=expected or response['Metadata']['sha256']
 archive=target/'inputs.tar.gz';h=hashlib.sha256()
 with archive.open('xb') as out:
  while chunk:=response['Body'].read(1024*1024):h.update(chunk);out.write(chunk)
 assert h.hexdigest()==expected,'Archive differs'
 with tarfile.open(archive) as package:
  members=package.getmembers()
  assert all(m.isfile() and not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts for m in members)
  package.extractall(target,members=members,filter='data')
 return h.hexdigest()
source=settings['raw_inputs'];raw_hash=unpack(source['bucket'],source['key'],raw,source['sha256'])
code_hash=unpack(settings['bucket'],settings['prefix']+'/inputs/host-code.tar.gz',code)
manifest=json.loads((raw/'fresh-input-manifest.json').read_text())
for relative,expected in manifest['inputs'].items():
 h=hashlib.sha256()
 with (raw/relative).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==expected,relative
for relative,expected in settings['code_sha256'].items():
 assert hashlib.sha256((code/relative).read_bytes()).hexdigest()==expected,relative
resolved={{}}
if {role!r}=='client':
 for label,host in [('bedrock','bedrock-runtime.ap-southeast-1.amazonaws.com'),('sagemaker_runtime','runtime.sagemaker.ap-southeast-1.amazonaws.com')]:
  ips=sorted({{row[4][0] for row in socket.getaddrinfo(host,443,family=socket.AF_INET)}})
  assert ips and set(ips)<=set(settings[label+'_private_ips']),(label,ips)
  resolved[label]=ips
with (base/'settings.json').open('x') as out:json.dump(settings,out,indent=2)
with (base/'ready.json').open('x') as out:json.dump({{'role':{role!r},'resource_id':settings['local_resource_id'],'raw_archive_sha256':raw_hash,'code_archive_sha256':code_hash,'private_dns':resolved,'old_outputs_imported':False}},out,indent=2)
print('INPUTS_CODE_AND_NETWORK_VERIFIED',settings['local_resource_id'],resolved,flush=True)
"""
        cmd=c.send_command(InstanceIds=[r[role+"_instance"]],DocumentName="AWS-RunShellScript",
            Parameters={"commands":["set -eu","cloud-init status --wait",
                "test -f /opt/pycon-isolated/bootstrap.json","export PYTHONDONTWRITEBYTECODE=1",
                r["python"]+" -u -c "+shlex.quote(body)],"executionTimeout":["1200"]},
            TimeoutSeconds=600,OutputS3BucketName=r["bucket"],OutputS3KeyPrefix=r["prefix"]+"/setup-logs")
        commands[role]=cmd["Command"]["CommandId"]
        with (directory / f"prepare-{role}-{uuid4().hex}.json").open("x") as out:
            json.dump({"command_id":commands[role],"instance":r[role+"_instance"]},out)
        print("PREPARATION",role,commands[role],flush=True)
    for role,command in commands.items():
        started=time.monotonic()
        while time.monotonic()-started<1500:
            try: result=c.get_command_invocation(CommandId=command,InstanceId=r[role+"_instance"])
            except c.exceptions.InvocationDoesNotExist:time.sleep(3);continue
            if result["Status"]=="Success":
                print("HOST_READY",role,result.get("StandardOutputContent","")[-1200:],flush=True);break
            if result["Status"] not in ("Pending","InProgress","Delayed"):
                raise RuntimeError(f"{role} setup {result['Status']}: {result.get('StandardErrorContent','')[-3000:]}")
            time.sleep(5)
        else:raise TimeoutError("Preparation did not finish: "+role)
    code="/opt/pycon-isolated/code/scripts/control_isolated_aws.py"
    cmd=c.send_command(InstanceIds=[r["client_instance"]],DocumentName="AWS-RunShellScript",Parameters={
        "commands":["set -eu","test -f /opt/pycon-isolated/ready.json",
            "systemd-run --unit=pycon-isolated-controller --property=Restart=no "
            "--property=StandardOutput=append:/opt/pycon-isolated/controller.log --property=StandardError=inherit "
            "--setenv=PYCON_FRESH_ROOT=/opt/pycon-isolated/input --setenv=PYTHONDONTWRITEBYTECODE=1 "
            "--setenv=PYCON_RESULT_BUCKET="+r["bucket"]+" --setenv=PYCON_RESULT_PREFIX="+r["prefix"]+"/benchmark "
            +r["python"]+" -u "+code+" /opt/pycon-isolated/settings.json"],"executionTimeout":["120"]},TimeoutSeconds=120)
    with (directory / "controller-launch.json").open("x") as out:
        json.dump({"command_id":cmd["Command"]["CommandId"],"client_instance":r["client_instance"],
            "service":"pycon-isolated-controller","benchmark_completion_verified":False},out,indent=2)
    print("CONTROLLER_LAUNCH_COMMAND",cmd["Command"]["CommandId"],flush=True)


if __name__=="__main__":main(sys.argv[1])
