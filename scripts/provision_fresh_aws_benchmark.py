"""Provision one isolated, three-hour AWS benchmark host; never resume old runs.

Every created resource is recorded in a new append-only local journal. This
script is deliberately not an idempotent resume command: do not rerun it after
a partial failure without inspecting the journal and the AWS resources first.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from uuid import uuid4
import argparse

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-southeast-1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-vpc-only", help="Existing provision name after failure immediately after VPC creation")
    args = parser.parse_args()
    session = boto3.Session(profile_name="admin-general", region_name=REGION)
    cfg = Config(connect_timeout=10, read_timeout=45, retries={"max_attempts": 2})
    ec2 = session.client("ec2", config=cfg)
    iam = session.client("iam", config=cfg)
    scheduler = session.client("scheduler", config=cfg)
    account = session.client("sts", config=cfg).get_caller_identity()["Account"]
    name = args.recover_vpc_only or f"pycon-fresh-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    directory = ROOT / "runs" / name
    if not args.recover_vpc_only:
        directory.mkdir(exist_ok=False)
    else:
        prior = list(directory.glob("*.json"))
        if len(prior) != 1 or json.loads(prior[0].read_text())["kind"] != "environment":
            raise RuntimeError("Recovery is restricted to failure immediately after VPC creation")
        previous = json.loads(prior[0].read_text())["value"]
        if previous["name"] != name or previous["account"] != account:
            raise RuntimeError("Recovery account/name mismatch")
    tags = [{"Key": "Name", "Value": name}, {"Key": "Project", "Value": "pycon-davao-model2vec"}]
    manifest = {"name": name, "account": account, "region": REGION,
                "fresh_outputs_only": True, "previous_outputs_reused": False,
                "instance_type": "c7i.4xlarge", "max_runtime_seconds": 10800,
                "credits_verified": False}

    def record(kind, value):
        manifest[kind] = value
        path = directory / f"resource-{uuid4().hex}.json"
        # Exclusive creation: no existing artifacts are overwritten.
        with path.open("x") as out:
            json.dump({"time": datetime.now(timezone.utc).isoformat(), "kind": kind,
                       "value": value}, out, indent=2)
        print(kind, json.dumps(value), flush=True)

    if not args.recover_vpc_only:
        record("environment", dict(manifest))
    offerings = ec2.describe_instance_type_offerings(
        LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": ["c7i.4xlarge"]}])
    zones = sorted(x["Location"] for x in offerings["InstanceTypeOfferings"])
    if not zones:
        raise RuntimeError("No c7i.4xlarge offering in Singapore")
    ami = session.client("ssm", config=cfg).get_parameter(
        Name="/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id")["Parameter"]["Value"]
    if args.recover_vpc_only:
        candidates = ec2.describe_vpcs(Filters=[{"Name": "tag:Name", "Values": [name]}])["Vpcs"]
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one previously created VPC")
        vpc = candidates[0]["VpcId"]
        if ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]:
            raise RuntimeError("VPC recovery must not duplicate existing subnets")
    else:
        vpc = ec2.create_vpc(CidrBlock="10.84.0.0/16", TagSpecifications=[{"ResourceType": "vpc", "Tags": tags}])["Vpc"]["VpcId"]
    record("vpc_id", vpc)
    ec2.modify_vpc_attribute(VpcId=vpc, EnableDnsSupport={"Value": True})
    ec2.modify_vpc_attribute(VpcId=vpc, EnableDnsHostnames={"Value": True})
    subnet = ec2.create_subnet(VpcId=vpc, CidrBlock="10.84.1.0/24", AvailabilityZone=zones[0],
        TagSpecifications=[{"ResourceType": "subnet", "Tags": tags}])["Subnet"]["SubnetId"]
    record("subnet_id", subnet)
    igw = ec2.create_internet_gateway(TagSpecifications=[{"ResourceType": "internet-gateway", "Tags": tags}])["InternetGateway"]["InternetGatewayId"]
    record("internet_gateway_id", igw)
    ec2.attach_internet_gateway(InternetGatewayId=igw, VpcId=vpc)
    routes = ec2.create_route_table(VpcId=vpc, TagSpecifications=[{"ResourceType": "route-table", "Tags": tags}])["RouteTable"]["RouteTableId"]
    record("route_table_id", routes)
    ec2.associate_route_table(RouteTableId=routes, SubnetId=subnet)
    ec2.create_route(RouteTableId=routes, DestinationCidrBlock="0.0.0.0/0", GatewayId=igw)
    host_sg = ec2.create_security_group(GroupName=name + "-host", Description="No inbound access; SSM managed benchmark", VpcId=vpc, TagSpecifications=[{"ResourceType": "security-group", "Tags": tags}])["GroupId"]
    record("host_security_group", host_sg)
    endpoint_sg = ec2.create_security_group(GroupName=name + "-bedrock", Description="HTTPS only from benchmark host", VpcId=vpc, TagSpecifications=[{"ResourceType": "security-group", "Tags": tags}])["GroupId"]
    record("endpoint_security_group", endpoint_sg)
    ec2.authorize_security_group_ingress(GroupId=endpoint_sg, IpPermissions=[{
        "IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "UserIdGroupPairs": [{"GroupId": host_sg}]}])
    endpoint = ec2.create_vpc_endpoint(VpcId=vpc, VpcEndpointType="Interface",
        ServiceName=f"com.amazonaws.{REGION}.bedrock-runtime", SubnetIds=[subnet],
        SecurityGroupIds=[endpoint_sg], PrivateDnsEnabled=True,
        TagSpecifications=[{"ResourceType": "vpc-endpoint", "Tags": tags}])["VpcEndpoint"]
    record("bedrock_endpoint_id", endpoint["VpcEndpointId"])
    s3_endpoint = ec2.create_vpc_endpoint(VpcId=vpc, VpcEndpointType="Gateway",
        ServiceName=f"com.amazonaws.{REGION}.s3", RouteTableIds=[routes],
        TagSpecifications=[{"ResourceType": "vpc-endpoint", "Tags": tags}])["VpcEndpoint"]["VpcEndpointId"]
    record("s3_endpoint_id", s3_endpoint)
    role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}), Tags=tags)["Role"]
    record("instance_role_arn", role["Arn"])
    iam.attach_role_policy(RoleName=name, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    bucket = f"sagemaker-{REGION}-{account}"
    prefix = f"pycon-davao-model2vec/{name}"
    record("s3_prefix", f"s3://{bucket}/{prefix}/")
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*"},
        {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}", "Condition": {"StringLike": {"s3:prefix": [prefix + "/*"]}}},
        {"Effect": "Allow", "Action": "bedrock:InvokeModel", "Resource": [
            f"arn:aws:bedrock:{REGION}::foundation-model/cohere.embed-english-v3",
            "arn:aws:bedrock:*::foundation-model/cohere.embed-v4*",
            f"arn:aws:bedrock:{REGION}:{account}:inference-profile/global.cohere.embed-v4:0"]}]}
    iam.put_role_policy(RoleName=name, PolicyName="fresh-benchmark-only", PolicyDocument=json.dumps(policy))
    iam.create_instance_profile(InstanceProfileName=name, Tags=tags)
    iam.add_role_to_instance_profile(InstanceProfileName=name, RoleName=name)
    record("instance_profile", name)
    schedule_role = iam.create_role(RoleName=name + "-stop", AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "scheduler.amazonaws.com"}, "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}), Tags=tags)["Role"]
    record("scheduler_role_arn", schedule_role["Arn"])
    # Allow time for the new instance profile to propagate before EC2 launch.
    time.sleep(15)
    user_data = """#!/bin/bash
set -euxo pipefail
# Schedule before installing anything; this remains effective on setup failure.
shutdown -h +180
install -d -m 0755 /opt/pycon-fresh
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip unzip
python3 -m venv /opt/pycon-fresh/venv
/opt/pycon-fresh/venv/bin/pip install boto3
date -u > /opt/pycon-fresh/bootstrap-ready.txt
"""
    instance = ec2.run_instances(ImageId=ami, InstanceType="c7i.4xlarge", MinCount=1, MaxCount=1,
        ClientToken=name, IamInstanceProfile={"Name": name},
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet, "Groups": [host_sg], "AssociatePublicIpAddress": True}],
        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
        InstanceInitiatedShutdownBehavior="stop", UserData=user_data,
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 80, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": False}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": tags}, {"ResourceType": "volume", "Tags": tags}])["Instances"][0]
    record("instance_id", instance["InstanceId"])
    stop_at = datetime.now(timezone.utc) + timedelta(hours=3)
    iam.put_role_policy(RoleName=name + "-stop", PolicyName="stop-one-host", PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "ec2:StopInstances", "Resource": f"arn:aws:ec2:{REGION}:{account}:instance/{instance['InstanceId']}"}]}))
    time.sleep(10)
    scheduled = scheduler.create_schedule(Name=name + "-stop", ScheduleExpression=f"at({stop_at:%Y-%m-%dT%H:%M:%S})",
        ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"},
        Target={"Arn": "arn:aws:scheduler:::aws-sdk:ec2:stopInstances", "RoleArn": schedule_role["Arn"],
                "Input": json.dumps({"InstanceIds": [instance["InstanceId"]]}),
                "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2}},
        State="ENABLED", ClientToken=name)
    record("automatic_stop", {"utc": stop_at.isoformat(), "schedule_arn": scheduled["ScheduleArn"]})
    record("ami", ami)
    with (directory / "resources.json").open("x") as out:
        json.dump(manifest, out, indent=2)
    print("RESOURCE_MANIFEST", directory / "resources.json", flush=True)


if __name__ == "__main__":
    main()
