"""User-authorized halt: quiesce, snapshot exact experiment disks, stop compute.

No EBS volumes, S3 objects, local files, model/config definitions or reports are
deleted. Only the ephemeral paid SageMaker endpoint is removed after metadata
is preserved. No resources may be restarted by this script.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time
from uuid import uuid4
import boto3
from botocore.config import Config


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'runs/pycon-isolated-20261002-192911-eb3adc'
    r=json.loads((folder/'resources.json').read_text())
    expected={r['client_instance'],r['server_instance']}
    expected.update(x['instance_id'] for x in json.loads((folder/'dedicated-resources.json').read_text())['models'].values())
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    cfg=Config(connect_timeout=5,read_timeout=30,retries={'max_attempts':0})
    ec,ssm,sm,s3=[session.client(n,config=cfg) for n in ('ec2','ssm','sagemaker','s3')]
    assert session.client('sts',config=cfg).get_caller_identity()['Account']==r['account']
    instances=[i for page in ec.get_paginator('describe_instances').paginate(Filters=[{'Name':'tag:Project','Values':['pycon-davao-model2vec']}]) for g in page['Reservations'] for i in g['Instances'] if i['State']['Name']!='terminated']
    assert {i['InstanceId'] for i in instances}==expected,'Unexpected project resources; inspect exact targets'
    marker=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8]
    local=folder/('preserved-stop-'+marker);local.mkdir(exist_ok=False)
    prefix=r['prefix']+'/preservation/'+marker
    def save(label,value):
        data=json.dumps(value,indent=2,default=str).encode()
        with (local/(label+'.json')).open('xb') as out:out.write(data)
        s3.put_object(Bucket=r['bucket'],Key=prefix+'/'+label+'.json',Body=data,
            IfNoneMatch='*',ServerSideEncryption='AES256')
    save('instance-inventory',instances)
    endpoint=sm.describe_endpoint(EndpointName=r['teacher_endpoint'])
    config=sm.describe_endpoint_config(EndpointConfigName=endpoint['EndpointConfigName'])
    models=[sm.describe_model(ModelName=v['ModelName']) for v in config['ProductionVariants']]
    save('gpu-endpoint-restore-record',{'endpoint':endpoint,'config':config,'models':models})
    save('deployment-records',{'original':r,'dedicated':json.loads((folder/'dedicated-resources.json').read_text()),
        'models_and_dataset_unchanged':True,'automatic_resume_authorized':False})
    commands={}
    body=f'''import subprocess,json,datetime,pathlib,boto3
lines=subprocess.check_output(['systemctl','list-units','--all','--type=service','--no-legend','--plain']).decode().splitlines()
units=[row.split()[0] for row in lines if row.split() and row.split()[0].startswith('pycon-')]
before={{}}
for unit in units:
 before[unit]=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecStart','-p','ExecMainStatus'],capture_output=True,text=True).stdout
# Controller processes must stop before independent workers.
units.sort(key=lambda name:0 if 'controller' in name else 1)
stops={{}}
for unit in units:
 reply=subprocess.run(['systemctl','stop',unit],capture_output=True,text=True,timeout=120)
 stops[unit]={{'returncode':reply.returncode,'stderr':reply.stderr}}
subprocess.run(['sync'],check=True)
remaining=[]
import psutil
for p in psutil.process_iter(['cmdline']):
 args=p.info['cmdline'] or []
 if any(a.endswith(('/run_isolated_aws.py','/isolated_model_http.py','/control_isolated_aws.py','/control_parallel_builds.py','/control_dedicated_models.py')) for a in args):remaining.append(p.pid)
assert not remaining,remaining
result={{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':before,'stops':stops,'benchmark_workers_remaining':remaining,'files_deleted':False,'sync_completed':True}}
p=pathlib.Path('/opt/pycon-isolated/preserved-stop-{marker}.json')
with p.open('x') as out:json.dump(result,out,indent=2)
print(json.dumps(result),flush=True)
'''
    # Client and Cohere first; all other dedicated hosts are already build-idle.
    order=sorted(instances,key=lambda x:0 if x['InstanceId']==r['client_instance'] else 1 if any(t.get('Value')=='cohere_v3' for t in x.get('Tags',[])) else 2)
    for host in order:
        if host['State']['Name']!='running':continue
        reply=ssm.send_command(InstanceIds=[host['InstanceId']],DocumentName='AWS-RunShellScript',
            Parameters={'commands':['PYTHONDONTWRITEBYTECODE=1 '+r['python']+' -u -c '+shlex.quote(body)],'executionTimeout':['240']},TimeoutSeconds=120)
        commands[host['InstanceId']]=reply['Command']['CommandId']
    save('quiesce-commands',commands)
    pending=dict(commands);limit=time.monotonic()+300
    while pending and time.monotonic()<limit:
        for instance,command in list(pending.items()):
            try:value=ssm.get_command_invocation(CommandId=command,InstanceId=instance)
            except ssm.exceptions.InvocationDoesNotExist:continue
            if value['Status']=='Success':
                save('quiesced-'+instance,value);pending.pop(instance)
                print('WORKERS_STOPPED_AND_FLUSHED',instance,flush=True)
            elif value['Status'] not in ('Pending','InProgress','Delayed'):
                save('quiesce-failure-'+instance,value)
                raise RuntimeError('Quiesce failed; disks preserved; inspect '+instance)
        if pending:time.sleep(2)
    if pending:raise RuntimeError('Quiesce incomplete; inspect before snapshot '+str(pending))
    snapshots=[]
    for host in instances:
        for disk in host.get('BlockDeviceMappings',[]):
            if 'Ebs' not in disk:continue
            volume=disk['Ebs']['VolumeId']
            tags=[{'Key':'Project','Value':'pycon-davao-model2vec'},{'Key':'Preservation','Value':marker},{'Key':'SourceInstance','Value':host['InstanceId']},{'Key':'Name','Value':'pycon-preserved-'+host['InstanceId']}]
            snap=ec.create_snapshot(VolumeId=volume,Description='User-requested preserved benchmark halt '+marker,
                TagSpecifications=[{'ResourceType':'snapshot','Tags':tags}])
            row={'instance_id':host['InstanceId'],'volume_id':volume,'snapshot_id':snap['SnapshotId'],'initial_state':snap['State']}
            snapshots.append(row);save('snapshot-'+host['InstanceId'],row)
            print('SNAPSHOT_REQUESTED',json.dumps(row),flush=True)
    save('snapshot-manifest',snapshots)
    # Stop, not terminate: source disks stay recoverable even while snapshots run.
    active=[i['InstanceId'] for i in instances if i['State']['Name'] in ('running','pending')]
    if active:
        result=ec.stop_instances(InstanceIds=active)
        save('ec2-stop-request',result)
        print('EC2_STOP_REQUESTED',active,flush=True)
    # Endpoint weights/config and vectors already have persistent records.
    sm.delete_endpoint(EndpointName=r['teacher_endpoint'])
    save('gpu-stop-request',{'endpoint':r['teacher_endpoint'],'endpoint_only_deleted':True,
        'model_and_endpoint_config_retained':True,'vectors_and_logs_retained':True})
    print('GPU_ENDPOINT_SHUTDOWN_REQUESTED',r['teacher_endpoint'],flush=True)
    save('halt-manifest',{'time_utc':datetime.now(timezone.utc).isoformat(),'instance_ids':sorted(expected),
        'snapshots':snapshots,'backup_s3':'s3://'+r['bucket']+'/'+prefix+'/',
        'existing_outputs_preserved':True,'automatic_restart_authorized':False,
        'verification_pending':True,'retained_storage_and_private_endpoints_still_charge':True})
    print('PRESERVATION_RECORD',local,flush=True)
    print('BACKUP_S3','s3://'+r['bucket']+'/'+prefix+'/',flush=True)


if __name__=='__main__':main()
