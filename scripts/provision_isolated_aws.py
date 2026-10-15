"""Provision separated CPU client/server and GPU endpoint with three-hour guards.

Run once per staged deployment. Journals use exclusive files; failures never
silently provision replacements. No deleted or rejected output is imported.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
import boto3
from botocore.config import Config


def main(directory):
    directory = Path(directory)
    staged = json.loads((directory / "deployment-inputs.json").read_text())
    with (directory / "provision-once.lock").open("x"):
        pass
    s = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0})
    if s.client("sts",config=cfg).get_caller_identity()["Account"] != staged["account"]:
        raise RuntimeError("Wrong AWS account")
    ec2, iam, sm, scheduler = [s.client(x,config=cfg) for x in ("ec2","iam","sagemaker","scheduler")]
    q=s.client("service-quotas",config=cfg).get_service_quota(ServiceCode="sagemaker",QuotaCode="L-B67CFA0C")
    if q["Quota"]["Value"] < 1: raise RuntimeError("No permitted g4dn GPU endpoint")
    standard=s.client("service-quotas",config=cfg).get_service_quota(ServiceCode="ec2",QuotaCode="L-1216C47A")["Quota"]["Value"]
    running=[x for page in ec2.get_paginator("describe_instances").paginate(Filters=[{"Name":"instance-state-name","Values":["running","pending"]}])
             for group in page["Reservations"] for x in group["Instances"]]
    types={x["InstanceType"] for x in running if x["InstanceType"][0] in "acdhimrtz"}
    sizes={x["InstanceType"]:x["VCpuInfo"]["DefaultVCpus"] for x in ec2.describe_instance_types(InstanceTypes=sorted(types))["InstanceTypes"]} if types else {}
    if sum(sizes.get(x["InstanceType"],0) for x in running)+24 > standard:
        raise RuntimeError("Insufficient currently free standard EC2 vCPU allowance for both CPU hosts")
    name=staged["name"]; account=staged["account"]; region=staged["region"]
    bucket=staged["bucket"]; prefix=staged["prefix"]
    vpc="vpc-03e333847a7780ea2"; subnet="subnet-04f20ff6c937cb1b9"
    endpoint_sg="sg-0ed3d905220df50b5"; bedrock_endpoint="vpce-058d30102ee2844d0"
    if ec2.describe_subnets(SubnetIds=[subnet])["Subnets"][0]["VpcId"] != vpc:
        raise RuntimeError("Project subnet changed")
    record={**staged,"vpc_id":vpc,"subnet_id":subnet,"teacher_endpoint":name+"-e5",
        "python":"/opt/pycon-isolated/venv/bin/python",
        "output_directory":"/opt/pycon-isolated/benchmark","bedrock_endpoint_id":bedrock_endpoint}
    def journal(kind,value):
        record[kind]=value
        with (directory / f"resource-{kind}-{uuid4().hex}.json").open("x") as out:
            json.dump({"time":datetime.now(timezone.utc).isoformat(),"kind":kind,"value":value},out,indent=2)
        print(kind,json.dumps(value),flush=True)
    tags=[{"Key":"Name","Value":name},{"Key":"Project","Value":"pycon-davao-model2vec"}]
    def sg(role):
        return ec2.create_security_group(GroupName=name+"-"+role,Description="Isolated benchmark "+role,
            VpcId=vpc,TagSpecifications=[{"ResourceType":"security-group","Tags":tags}])["GroupId"]
    client_sg=sg("client");journal("client_sg",client_sg)
    server_sg=sg("server");journal("server_sg",server_sg)
    ec2.authorize_security_group_ingress(GroupId=server_sg,IpPermissions=[{"IpProtocol":"tcp",
        "FromPort":8080,"ToPort":8080,"UserIdGroupPairs":[{"GroupId":client_sg}]}])
    ec2.authorize_security_group_ingress(GroupId=endpoint_sg,IpPermissions=[{"IpProtocol":"tcp",
        "FromPort":443,"ToPort":443,"UserIdGroupPairs":[{"GroupId":client_sg}]}])
    vpce=ec2.create_vpc_endpoint(VpcId=vpc,VpcEndpointType="Interface",
        ServiceName=f"com.amazonaws.{region}.sagemaker.runtime",SubnetIds=[subnet],
        SecurityGroupIds=[endpoint_sg],PrivateDnsEnabled=True,
        TagSpecifications=[{"ResourceType":"vpc-endpoint","Tags":tags}])["VpcEndpoint"]
    journal("sagemaker_runtime_endpoint_id",vpce["VpcEndpointId"])
    trust={"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{
        "Service":["ec2.amazonaws.com","sagemaker.amazonaws.com"]},"Action":"sts:AssumeRole"}]}
    role=iam.create_role(RoleName=name,AssumeRolePolicyDocument=json.dumps(trust),Tags=tags)["Role"]
    journal("role_arn",role["Arn"])
    iam.attach_role_policy(RoleName=name,PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    raw=staged["raw_inputs"]
    policy={"Version":"2012-10-17","Statement":[
        {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":f"arn:aws:s3:::{bucket}/{prefix}/*"},
        {"Effect":"Allow","Action":"s3:GetObject","Resource":f"arn:aws:s3:::{raw['bucket']}/{raw['key']}"},
        {"Effect":"Allow","Action":"s3:ListBucket","Resource":f"arn:aws:s3:::{bucket}","Condition":{"StringLike":{"s3:prefix":[prefix+"/*"]}}},
        {"Effect":"Allow","Action":"sagemaker:InvokeEndpoint","Resource":f"arn:aws:sagemaker:{region}:{account}:endpoint/{name}-e5"},
        {"Effect":"Allow","Action":"sagemaker:DescribeEndpoint","Resource":f"arn:aws:sagemaker:{region}:{account}:endpoint/{name}-e5"},
        {"Effect":"Allow","Action":"bedrock:InvokeModel","Resource":f"arn:aws:bedrock:{region}::foundation-model/cohere.embed-english-v3"},
        {"Effect":"Allow","Action":["ecr:GetAuthorizationToken","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer","logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"},
        {"Effect":"Allow","Action":"ssm:GetCommandInvocation","Resource":"*"}]}
    iam.put_role_policy(RoleName=name,PolicyName="isolated-benchmark",PolicyDocument=json.dumps(policy))
    iam.create_instance_profile(InstanceProfileName=name,Tags=tags)
    iam.add_role_to_instance_profile(InstanceProfileName=name,RoleName=name)
    guard=iam.create_role(RoleName=name+"-cap",AssumeRolePolicyDocument=json.dumps({"Version":"2012-10-17","Statement":[{
        "Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole",
        "Condition":{"StringEquals":{"aws:SourceAccount":account}}}]}),Tags=tags)["Role"]
    journal("guard_role",guard["Arn"])
    def guard_policy(ids):
        stm=[{"Effect":"Allow","Action":"sagemaker:DeleteEndpoint","Resource":f"arn:aws:sagemaker:{region}:{account}:endpoint/{name}-e5"}]
        if ids: stm.append({"Effect":"Allow","Action":"ec2:StopInstances","Resource":[f"arn:aws:ec2:{region}:{account}:instance/{i}" for i in ids]})
        iam.put_role_policy(RoleName=name+"-cap",PolicyName="stop-exact-deployment",PolicyDocument=json.dumps({"Version":"2012-10-17","Statement":stm}))
    guard_policy([]);time.sleep(15)
    def schedule(suffix,action,payload):
        until=datetime.now(timezone.utc)+timedelta(seconds=10800)
        response=scheduler.create_schedule(Name=name+"-"+suffix,ScheduleExpression=f"at({until:%Y-%m-%dT%H:%M:%S})",
            ScheduleExpressionTimezone="UTC",FlexibleTimeWindow={"Mode":"OFF"},State="ENABLED",
            Target={"Arn":"arn:aws:scheduler:::aws-sdk:"+action,"RoleArn":guard["Arn"],"Input":json.dumps(payload),
                    "RetryPolicy":{"MaximumEventAgeInSeconds":300,"MaximumRetryAttempts":2}})
        journal(suffix+"_guard",{"utc":until.isoformat(),"arn":response["ScheduleArn"]})
    # Endpoint guard exists before any GPU endpoint is requested.
    schedule("gpu-stop","sagemaker:deleteEndpoint",{"EndpointName":name+"-e5"})
    sm.create_model(ModelName=name+"-e5",ExecutionRoleArn=role["Arn"],PrimaryContainer={
        "Image":staged["gpu_serving_image"],"ModelDataUrl":f"s3://{bucket}/{prefix}/inputs/teacher-code.tar.gz",
        "Environment":{"SAGEMAKER_PROGRAM":"inference.py","SAGEMAKER_SUBMIT_DIRECTORY":"/opt/ml/model/code",
            "SAGEMAKER_MODEL_SERVER_WORKERS":"1","MMS_DEFAULT_RESPONSE_TIMEOUT":"120",
            "PYCON_RESOURCE_ID":"sagemaker:"+name+"-e5","PYTHONDONTWRITEBYTECODE":"1"}},Tags=tags)
    journal("sagemaker_model",name+"-e5")
    sm.create_endpoint_config(EndpointConfigName=name+"-e5",ProductionVariants=[{"VariantName":"only-teacher",
        "ModelName":name+"-e5","InitialInstanceCount":1,"InstanceType":"ml.g4dn.xlarge",
        "InitialVariantWeight":1,"ModelDataDownloadTimeoutInSeconds":900,
        "ContainerStartupHealthCheckTimeoutInSeconds":900}],Tags=tags)
    sm.create_endpoint(EndpointName=name+"-e5",EndpointConfigName=name+"-e5",Tags=tags)
    journal("gpu_endpoint_requested",name+"-e5")
    ami=s.client("ssm",config=cfg).get_parameter(Name="/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id")["Parameter"]["Value"]
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
    for kind,typ,group in (("client","c7i.2xlarge",client_sg),("server","c7i.4xlarge",server_sg)):
        t=tags+[{"Key":"Workload","Value":kind}]
        host=ec2.run_instances(ImageId=ami,InstanceType=typ,MinCount=1,MaxCount=1,
            ClientToken=name+"-"+kind,IamInstanceProfile={"Name":name},UserData=bootstrap,
            NetworkInterfaces=[{"DeviceIndex":0,"SubnetId":subnet,"Groups":[group],"AssociatePublicIpAddress":True}],
            MetadataOptions={"HttpTokens":"required","HttpEndpoint":"enabled"},InstanceInitiatedShutdownBehavior="stop",
            BlockDeviceMappings=[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3","Encrypted":True,"DeleteOnTermination":False}}],
            TagSpecifications=[{"ResourceType":"instance","Tags":t},{"ResourceType":"volume","Tags":t}])["Instances"][0]
        ids.append(host["InstanceId"]);journal(kind+"_instance",host["InstanceId"])
        journal(kind+"_private_ip",host["PrivateIpAddress"])
        guard_policy(ids);schedule(kind+"-stop","ec2:stopInstances",{"InstanceIds":[host["InstanceId"]]})
    policy["Statement"].append({"Effect":"Allow","Action":"ssm:SendCommand","Resource":[
        f"arn:aws:ssm:{region}::document/AWS-RunShellScript",f"arn:aws:ec2:{region}:{account}:instance/{record['server_instance']}"]})
    iam.put_role_policy(RoleName=name,PolicyName="isolated-benchmark",PolicyDocument=json.dumps(policy))
    for label,endpoint in (("bedrock",bedrock_endpoint),("sagemaker_runtime",vpce["VpcEndpointId"])):
        row=ec2.describe_vpc_endpoints(VpcEndpointIds=[endpoint])["VpcEndpoints"][0]
        interfaces=ec2.describe_network_interfaces(NetworkInterfaceIds=row["NetworkInterfaceIds"])["NetworkInterfaces"]
        journal(label+"_private_ips",[r["PrivateIpAddress"] for r in interfaces])
    journal("compute_requested",True)
    with (directory / "resources.json").open("x") as out: json.dump(record,out,indent=2)
    print("RESOURCES",directory / "resources.json",flush=True)


if __name__ == "__main__": main(sys.argv[1])
