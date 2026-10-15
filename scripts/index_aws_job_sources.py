"""Index original SageMaker source objects; add provenance tags, never overwrite."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse,quote
from uuid import uuid4
import boto3


def main():
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    sm,s3=session.client('sagemaker'),session.client('s3')
    bucket='sagemaker-ap-southeast-1-<aws-account-id>'
    index_prefix='pycon-davao-model2vec/demo-source-index-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6]
    jobs=[]
    sources={}
    for page in sm.get_paginator('list_training_jobs').paginate(NameContains='pycon-'):
        for summary in page['TrainingJobSummaries']:
            name=summary['TrainingJobName']
            job=sm.describe_training_job(TrainingJobName=name)
            item={'job_name':name,'status_at_index':job['TrainingJobStatus'],
                'job_arn':job['TrainingJobArn'],'algorithm':job['AlgorithmSpecification'],'sources':[]}
            for channel in job.get('InputDataConfig',[]):
                uri=urlparse(channel['DataSource']['S3DataSource']['S3Uri'])
                for listing in s3.get_paginator('list_objects_v2').paginate(Bucket=uri.netloc,Prefix=uri.path.lstrip('/')):
                    for obj in listing.get('Contents',[]):
                        if not (obj['Key'].endswith('.py') or obj['Key'].endswith('.json')):
                            continue
                        if obj['Size']>2**20:
                            continue
                        data=s3.get_object(Bucket=uri.netloc,Key=obj['Key'])['Body'].read()
                        checksum=sha256(data).hexdigest()
                        filename=obj['Key'].rsplit('/',1)[-1]
                        local='code/'+name+'/'+channel['ChannelName']+'/'+filename
                        sources[local]=data.decode('utf-8')
                        item['sources'].append({'filename':filename,'channel':channel['ChannelName'],
                            'bytes':len(data),'sha256':checksum,'local_relative_path':local,
                            's3_uri':'s3://'+uri.netloc+'/'+obj['Key'],
                            'console_url':'https://s3.console.aws.amazon.com/s3/object/'+uri.netloc+
                                '?region=ap-southeast-1&prefix='+quote(obj['Key'],safe='')})
            invoked=job['AlgorithmSpecification'].get('ContainerArguments',[])
            if not any(s['filename'].endswith('.py') and any(s['filename'] in a for a in invoked) for s in item['sources']):
                raise ValueError('Actual worker entrypoint not found in original inputs: '+name)
            payload=(json.dumps(item,sort_keys=True,indent=2)+'\n').encode()
            key=index_prefix+'/'+name+'/manifest.json'
            s3.put_object(Bucket=bucket,Key=key,Body=payload,IfNoneMatch='*',
                Metadata={'sha256':sha256(payload).hexdigest()})
            item['source_manifest_s3_uri']='s3://'+bucket+'/'+key
            tags=sm.list_tags(ResourceArn=job['TrainingJobArn'])['Tags']
            if not any(t['Key']=='DemoSourceManifest' for t in tags):
                sm.add_tags(ResourceArn=job['TrainingJobArn'],Tags=[{'Key':'DemoSourceManifest',
                            'Value':item['source_manifest_s3_uri']}])
                item['demo_source_tag_added']=True
            else:
                item['demo_source_tag_added']=False
                item['existing_demo_source_tag_preserved']=True
            jobs.append(item)
    catalog={'region':'ap-southeast-1','profile':'admin-general','jobs':jobs,
        's3_index':'s3://'+bucket+'/'+index_prefix+'/',
        'scope':'SageMaker jobs; historical EC2 benchmark source is separate',
        'no_compute_launched':True,'no_existing_source_changed':True}
    data=(json.dumps(catalog,sort_keys=True,indent=2)+'\n').encode()
    s3.put_object(Bucket=bucket,Key=index_prefix+'/index.json',Body=data,IfNoneMatch='*',
        Metadata={'sha256':sha256(data).hexdigest()})
    print(json.dumps({'catalog':catalog,'original_source_text':sources}),flush=True)


if __name__=='__main__':
    main()
