"""Provision four exclusive static hosts and one exclusive Cohere client.

All new hosts retain the original earliest shutdown deadline. This does not
extend GPU lifetime, alter models, or replace existing benchmark data.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from uuid import uuid4
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def main(folder):
    folder = Path(folder)
    old = json.loads((folder / 'resources.json').read_text())
    session = boto3.Session(profile_name='admin-general', region_name=old['region'])
    cfg = Config(connect_timeout=5, read_timeout=30, retries={'max_attempts': 0})
    ec, iam, sched, ssm = [session.client(n, config=cfg) for n in ('ec2','iam','scheduler','ssm')]
    assert session.client('sts',config=cfg).get_caller_identity()['Account'] == old['account']
    deadline = datetime.fromisoformat(old['gpu-stop_guard']['utc'])
    seconds = int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert seconds > 1800, 'Insufficient unchanged runtime remaining'
    assert ec.describe_instances(InstanceIds=[old['server_instance']])['Reservations'][0]['Instances'][0]['State']['Name'] == 'stopped'
    allowance = session.client('service-quotas',config=cfg).get_service_quota(ServiceCode='ec2',QuotaCode='L-1216C47A')['Quota']['Value']
    rows = [i for p in ec.get_paginator('describe_instances').paginate(Filters=[{'Name':'instance-state-name','Values':['running','pending']}]) for g in p['Reservations'] for i in g['Instances']]
    types = {i['InstanceType'] for i in rows}
    cpus = {t['InstanceType']:t['VCpuInfo']['DefaultVCpus'] for t in ec.describe_instance_types(InstanceTypes=sorted(types))['InstanceTypes']} if types else {}
    existing = {i['ClientToken']:i for i in rows if i.get('ClientToken','').startswith(old['name']+'-dedicated-')}
    assert sum(cpus[i['InstanceType']] for i in rows)+18-sum(cpus[i['InstanceType']] for i in existing.values()) <= allowance
    if not (folder/'dedicated-provision-once.lock').exists():
        with (folder/'dedicated-provision-once.lock').open('x'): pass
    else:
        assert existing and not (folder/'dedicated-resources.json').exists(), 'Provisioning already complete or unexpected lock'
    ami = ssm.get_parameter(Name='/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id')['Parameter']['Value']
    record = {'original':old['name'], 'deadline_utc':deadline.isoformat(), 'models':{},
              'static_instance_type':'c7i.xlarge', 'static_vcpus':4,
              'cohere_instance_type':'c7i.large', 'previous_shared_server_stopped':True}
    for name in ('potion_retrieval','plain_student','custom_student','tokenlearn_custom_student','cohere_v3'):
        cohort = name != 'cohere_v3'
        token = old['name']+'-dedicated-'+('tokenlearn' if name=='tokenlearn_custom_student' else name)
        tags = [{'Key':'Project','Value':'pycon-davao-model2vec'}, {'Key':'Name','Value':old['name']+'-'+name}, {'Key':'ExclusiveModel','Value':name}]
        boot = f'''#!/bin/bash
set -euxo pipefail
shutdown -h +{max(1,seconds//60)}
install -d /opt/pycon-isolated
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
python3 -m venv /opt/pycon-isolated/venv
/opt/pycon-isolated/venv/bin/pip install boto3 numpy==2.5.3 model2vec==0.9.0 psutil==7.2.2 threadpoolctl
/opt/pycon-isolated/venv/bin/python -c 'import pathlib; pathlib.Path("/opt/pycon-isolated/bootstrap.json").open("x").write("{{}}")'
'''
        host = existing.get(token)
        if host is None:
            host = ec.run_instances(ImageId=ami, InstanceType='c7i.xlarge' if cohort else 'c7i.large',MinCount=1,MaxCount=1,
            ClientToken=token, IamInstanceProfile={'Name':old['name']},UserData=boot,
            NetworkInterfaces=[{'DeviceIndex':0,'SubnetId':old['subnet_id'],'Groups':[old['server_sg'] if cohort else old['client_sg']],'AssociatePublicIpAddress':True}],
            MetadataOptions={'HttpTokens':'required','HttpEndpoint':'enabled'},InstanceInitiatedShutdownBehavior='stop',
            BlockDeviceMappings=[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':80,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':False}}],
                TagSpecifications=[{'ResourceType':'instance','Tags':tags},{'ResourceType':'volume','Tags':tags}])['Instances'][0]
        row = {'instance_id':host['InstanceId'],'private_ip':host['PrivateIpAddress'],'model':name,'instance_type':host['InstanceType']}
        with (folder/f'dedicated-{name}-{uuid4().hex}.json').open('x') as out: json.dump(row,out,indent=2)
        arn = f"arn:aws:ec2:{old['region']}:{old['account']}:instance/{host['InstanceId']}"
        iam.put_role_policy(RoleName=old['name']+'-cap',PolicyName='stop-dedicated-'+name,
            PolicyDocument=json.dumps({'Version':'2012-10-17','Statement':[{'Effect':'Allow','Action':'ec2:StopInstances','Resource':arn}]}))
        label = old['name']+'-d'+str(len(record['models']))+'-stop'
        try:
            guard = sched.get_schedule(Name=label)
        except ClientError as error:
            if error.response['Error']['Code'] != 'ResourceNotFoundException': raise
            sched.create_schedule(Name=label, ScheduleExpression=f'at({deadline:%Y-%m-%dT%H:%M:%S})',ScheduleExpressionTimezone='UTC',
                FlexibleTimeWindow={'Mode':'OFF'},State='ENABLED',Target={'Arn':'arn:aws:scheduler:::aws-sdk:ec2:stopInstances',
                    'RoleArn':old['guard_role'],'Input':json.dumps({'InstanceIds':[host['InstanceId']]}),
                    'RetryPolicy':{'MaximumEventAgeInSeconds':300,'MaximumRetryAttempts':2}})
            guard = sched.get_schedule(Name=label)
        assert guard['State']=='ENABLED'
        assert json.loads(guard['Target']['Input'])['InstanceIds']==[host['InstanceId']]
        row['stop_schedule']=label
        record['models'][name]=row
        print('DEDICATED_INSTANCE',json.dumps(row),flush=True)
    arns = [f"arn:aws:ec2:{old['region']}:{old['account']}:instance/{x['instance_id']}" for x in record['models'].values()]
    iam.put_role_policy(RoleName=old['name'],PolicyName='dedicated-host-commands',PolicyDocument=json.dumps({'Version':'2012-10-17','Statement':[
        {'Effect':'Allow','Action':'ssm:SendCommand','Resource':[f"arn:aws:ssm:{old['region']}::document/AWS-RunShellScript",*arns]}]}))
    with (folder/'dedicated-resources.json').open('x') as out: json.dump(record,out,indent=2)
    print('ALL_FIVE_DEDICATED_INSTANCES_REQUESTED',flush=True)


if __name__ == '__main__': main(sys.argv[1])
