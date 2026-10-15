"""Prepare exclusive model hosts; immediately launch Cohere on its own EC2."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
from uuid import uuid4
import boto3
from botocore.config import Config


def main(folder):
    folder = Path(folder)
    r = json.loads((folder/'resources.json').read_text())
    dedicated = json.loads((folder/'dedicated-resources.json').read_text())
    s = boto3.Session(profile_name='admin-general',region_name=r['region'])
    c = s.client('ssm',config=Config(connect_timeout=5,read_timeout=30,retries={'max_attempts':0}))
    commands = {}
    # Prioritize the paid API build; statics bootstrap independently.
    for name in ('cohere_v3','potion_retrieval','plain_student','custom_student','tokenlearn_custom_student'):
        row = dedicated['models'][name]
        settings = {**r, 'server_instance':row['instance_id'], 'server_private_ip':row['private_ip'],
                    'local_resource_id':row['instance_id'],
                    'prefix':r['prefix']+'/dedicated', 'output_directory':'/opt/pycon-isolated/dedicated-benchmark'}
        if name=='cohere_v3': settings['client_instance']=row['instance_id']
        body = f'''import boto3,hashlib,json,pathlib,socket,tarfile,subprocess
settings=json.loads({json.dumps(settings)!r});name={name!r}
base=pathlib.Path('/opt/pycon-isolated');raw=base/'input';code=base/'code'
raw.mkdir(exist_ok=False);code.mkdir(exist_ok=False)
s=boto3.client('s3',region_name='ap-southeast-1')
def unpack(bucket,key,target,expected=None):
 response=s.get_object(Bucket=bucket,Key=key);expected=expected or response['Metadata']['sha256']
 archive=target/'inputs.tar.gz';h=hashlib.sha256()
 with archive.open('xb') as out:
  while chunk:=response['Body'].read(1024*1024):h.update(chunk);out.write(chunk)
 assert h.hexdigest()==expected
 with tarfile.open(archive) as package:
  members=package.getmembers()
  assert all(m.isfile() and not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts for m in members)
  package.extractall(target,members=members,filter='data')
raw_input=settings['raw_inputs'];unpack(raw_input['bucket'],raw_input['key'],raw,raw_input['sha256'])
unpack(settings['bucket'],{r['prefix']!r}+'/inputs/host-code.tar.gz',code)
manifest=json.loads((raw/'fresh-input-manifest.json').read_text())
for relative,expected in manifest['inputs'].items():
 h=hashlib.sha256()
 with (raw/relative).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==expected,relative
for relative,expected in settings['code_sha256'].items():
 assert hashlib.sha256((code/relative).read_bytes()).hexdigest()==expected,relative
with (base/'settings.json').open('x') as out:json.dump(settings,out,indent=2)
with (base/'ready.json').open('x') as out:json.dump({{'model':name,'instance_id':settings['local_resource_id'],'weights_verified':True}},out)
if name=='cohere_v3':
 host='bedrock-runtime.ap-southeast-1.amazonaws.com'
 ips={{x[4][0] for x in socket.getaddrinfo(host,443,family=socket.AF_INET)}}
 assert ips and ips<=set(settings['bedrock_private_ips']),ips
 directory=pathlib.Path(settings['output_directory']);directory.mkdir(exist_ok=False)
 for filename in ('shared-catalog.json','shared-catalog.sha256.json','timing-queries.json'):
  reply=s.get_object(Bucket=settings['bucket'],Key={r['prefix']!r}+'/benchmark/'+filename)
  data=reply['Body'].read();assert hashlib.sha256(data).hexdigest()==reply['Metadata']['sha256']
  with (directory/filename).open('xb') as out:out.write(data)
 cmd=['systemd-run','--unit=pycon-dedicated-cohere-build','--property=Restart=no','--property=StandardOutput=append:/opt/pycon-isolated/dedicated-cohere-build.log','--property=StandardError=inherit','--setenv=PYCON_FRESH_ROOT='+str(raw),'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYCON_RESULT_BUCKET='+settings['bucket'],'--setenv=PYCON_RESULT_PREFIX='+settings['prefix']+'/benchmark',settings['python'],'-u',str(code/'scripts/run_isolated_aws.py'),'--directory',str(directory),'--settings',str(base/'settings.json'),'--phase','build','--model',name]
 subprocess.run(cmd,check=True)
print('DEDICATED_MODEL_READY',name,settings['local_resource_id'],flush=True)
'''
        response = c.send_command(InstanceIds=[row['instance_id']],DocumentName='AWS-RunShellScript',
            Parameters={'commands':['set -eu','cloud-init status --wait','test -f /opt/pycon-isolated/bootstrap.json',
                'PYTHONDONTWRITEBYTECODE=1 '+r['python']+' -u -c '+shlex.quote(body)],'executionTimeout':['1200']},
            TimeoutSeconds=600,OutputS3BucketName=r['bucket'],OutputS3KeyPrefix=r['prefix']+'/dedicated/setup-logs')
        commands[name] = {'instance_id':row['instance_id'],'command_id':response['Command']['CommandId']}
        print('PREPARING',name,json.dumps(commands[name]),flush=True)
    with (folder/f'dedicated-prepare-{uuid4().hex}.json').open('x') as out:json.dump(commands,out,indent=2)


if __name__=='__main__':main(sys.argv[1])
