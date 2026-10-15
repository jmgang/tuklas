"""Finish the verified pre-compute VolumeSize validation failure, once.

No existing resource is recreated or overwritten; GPU deadline stays unchanged.
"""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys,time
from uuid import uuid4
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def main(directory):
    directory=Path(directory);staged=json.loads((directory/'deployment-inputs.json').read_text())
    name=staged['name'];account=staged['account'];region=staged['region']
    if name!='pycon-isolated-20261002-192911-eb3adc':raise ValueError('Recovery is exact-target only')
    previous={}
    for p in directory.glob('resource-*.json'):
        row=json.loads(p.read_text());previous[row['kind']]=row['value']
    required={'client_sg','server_sg','sagemaker_runtime_endpoint_id','role_arn','guard_role','gpu-stop_guard','sagemaker_model'}
    if set(previous)!=required:raise RuntimeError('Unexpected provisioning stage; inspect instead of resuming')
    s=boto3.Session(profile_name='admin-general',region_name=region)
    cfg=Config(connect_timeout=5,read_timeout=30,retries={'max_attempts':0})
    ec2,iam,sm,scheduler=[s.client(x,config=cfg) for x in ('ec2','iam','sagemaker','scheduler')]
    if s.client('sts',config=cfg).get_caller_identity()['Account']!=account:raise ValueError('Wrong account')
    model=sm.describe_model(ModelName=name+'-e5')
    if model['PrimaryContainer']['Image']!=staged['gpu_serving_image'] or model['ExecutionRoleArn']!=previous['role_arn']:
        raise ValueError('Staged GPU model changed')
    for api,arg in ((sm.describe_endpoint,'EndpointName'),(sm.describe_endpoint_config,'EndpointConfigName')):
        try:api(**{arg:name+'-e5'})
        except ClientError as error:
            if error.response['Error']['Code']!='ValidationException' or 'Could not find' not in error.response['Error']['Message']:
                raise
        else:raise RuntimeError('GPU endpoint/config already exists; do not duplicate')
    existing=ec2.describe_instances(Filters=[{'Name':'tag:Name','Values':[name]}])['Reservations']
    if any(g['Instances'] for g in existing):raise RuntimeError('CPU host already exists; do not duplicate')
    deadline=datetime.fromisoformat(previous['gpu-stop_guard']['utc'])
    if (deadline-datetime.now(timezone.utc)).total_seconds()<1800:
        raise RuntimeError('Not enough time before unchanged GPU deadline; require user direction')
    with (directory/'finish-provision-once.lock').open('x'):pass
    record={**staged,**previous,'vpc_id':'vpc-03e333847a7780ea2','subnet_id':'subnet-04f20ff6c937cb1b9',
        'teacher_endpoint':name+'-e5','python':'/opt/pycon-isolated/venv/bin/python',
        'output_directory':'/opt/pycon-isolated/benchmark','bedrock_endpoint_id':'vpce-058d30102ee2844d0',
        'gpu_deadline_extended':False,'repair':'omit unsupported VolumeSizeInGB; keep local NVMe default'}
    def journal(kind,value):
        record[kind]=value
        with (directory/f'resource-{kind}-{uuid4().hex}.json').open('x') as out:
            json.dump({'time':datetime.now(timezone.utc).isoformat(),'kind':kind,'value':value},out,indent=2)
        print(kind,json.dumps(value),flush=True)
    tags=[{'Key':'Name','Value':name},{'Key':'Project','Value':'pycon-davao-model2vec'}]
    sm.create_endpoint_config(EndpointConfigName=name+'-e5',ProductionVariants=[{'VariantName':'only-teacher',
        'ModelName':name+'-e5','InitialInstanceCount':1,'InstanceType':'ml.g4dn.xlarge',
        'InitialVariantWeight':1,'ModelDataDownloadTimeoutInSeconds':900,'ContainerStartupHealthCheckTimeoutInSeconds':900}],Tags=tags)
    sm.create_endpoint(EndpointName=name+'-e5',EndpointConfigName=name+'-e5',Tags=tags)
    journal('gpu_endpoint_requested',name+'-e5')
    ami=s.client('ssm',config=cfg).get_parameter(Name='/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id')['Parameter']['Value']
    bootstrap="""#!/bin/bash
set -euxo pipefail
shutdown -h +180
install -d /opt/pycon-isolated
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
python3 -m venv /opt/pycon-isolated/venv
/opt/pycon-isolated/venv/bin/pip install boto3 numpy==2.5.3 model2vec==0.9.0 psutil==7.2.2 threadpoolctl
/opt/pycon-isolated/venv/bin/python -c 'import json,pathlib; pathlib.Path("/opt/pycon-isolated/bootstrap.json").open("x").write(json.dumps({"bootstrap":"ready"}))'
"""
    ids=[]
    for role,typ in (('client','c7i.2xlarge'),('server','c7i.4xlarge')):
        t=tags+[{'Key':'Workload','Value':role}]
        host=ec2.run_instances(ImageId=ami,InstanceType=typ,MinCount=1,MaxCount=1,ClientToken=name+'-'+role,
            IamInstanceProfile={'Name':name},UserData=bootstrap,
            NetworkInterfaces=[{'DeviceIndex':0,'SubnetId':record['subnet_id'],'Groups':[previous[role+'_sg']],'AssociatePublicIpAddress':True}],
            MetadataOptions={'HttpTokens':'required','HttpEndpoint':'enabled'},InstanceInitiatedShutdownBehavior='stop',
            BlockDeviceMappings=[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':80,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':False}}],
            TagSpecifications=[{'ResourceType':'instance','Tags':t},{'ResourceType':'volume','Tags':t}])['Instances'][0]
        ids.append(host['InstanceId']);journal(role+'_instance',host['InstanceId']);journal(role+'_private_ip',host['PrivateIpAddress'])
        iam.put_role_policy(RoleName=name+'-cap',PolicyName='stop-exact-deployment',PolicyDocument=json.dumps({'Version':'2012-10-17','Statement':[
            {'Effect':'Allow','Action':'sagemaker:DeleteEndpoint','Resource':f'arn:aws:sagemaker:{region}:{account}:endpoint/{name}-e5'},
            {'Effect':'Allow','Action':'ec2:StopInstances','Resource':[f'arn:aws:ec2:{region}:{account}:instance/{i}' for i in ids]}]}))
        until=datetime.now(timezone.utc)+timedelta(seconds=10800)
        response=scheduler.create_schedule(Name=name+'-'+role+'-stop',ScheduleExpression=f'at({until:%Y-%m-%dT%H:%M:%S})',
            ScheduleExpressionTimezone='UTC',FlexibleTimeWindow={'Mode':'OFF'},State='ENABLED',Target={
                'Arn':'arn:aws:scheduler:::aws-sdk:ec2:stopInstances','RoleArn':previous['guard_role'],
                'Input':json.dumps({'InstanceIds':[host['InstanceId']]}),
                'RetryPolicy':{'MaximumEventAgeInSeconds':300,'MaximumRetryAttempts':2}})
        journal(role+'-stop_guard',{'utc':until.isoformat(),'arn':response['ScheduleArn']})
    policy=json.loads(iam.get_role_policy(RoleName=name,PolicyName='isolated-benchmark')['PolicyDocument']
        if isinstance(iam.get_role_policy(RoleName=name,PolicyName='isolated-benchmark')['PolicyDocument'],str)
        else json.dumps(iam.get_role_policy(RoleName=name,PolicyName='isolated-benchmark')['PolicyDocument']))
    policy['Statement'].append({'Effect':'Allow','Action':'ssm:SendCommand','Resource':[
        f'arn:aws:ssm:{region}::document/AWS-RunShellScript',f"arn:aws:ec2:{region}:{account}:instance/{record['server_instance']}"]})
    iam.put_role_policy(RoleName=name,PolicyName='isolated-benchmark',PolicyDocument=json.dumps(policy))
    for label,endpoint in (('bedrock',record['bedrock_endpoint_id']),('sagemaker_runtime',record['sagemaker_runtime_endpoint_id'])):
        row=ec2.describe_vpc_endpoints(VpcEndpointIds=[endpoint])['VpcEndpoints'][0]
        interfaces=ec2.describe_network_interfaces(NetworkInterfaceIds=row['NetworkInterfaceIds'])['NetworkInterfaces']
        journal(label+'_private_ips',[r['PrivateIpAddress'] for r in interfaces])
    journal('compute_requested',True)
    with (directory/'resources.json').open('x') as out:json.dump(record,out,indent=2)
    print('RESOURCES',directory/'resources.json',flush=True)


if __name__=='__main__':main(sys.argv[1])
