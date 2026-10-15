#!/usr/bin/env python3
"""Explicit --launch required. One quality-only job; no latency, endpoints or EC2."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4
import boto3

ROOT=Path(__file__).resolve().parents[1]
import start_runtime_comparison_v2 as prior
MAX_RUNTIME_SECONDS=5000
TRAINING_USD_PER_HOUR=2.106


def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def stream_hash(body):
    h=sha256()
    for block in iter(lambda:body.read(4<<20),b''): h.update(block)
    body.close()
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--launch',action='store_true'); args=parser.parse_args()
    session=boto3.Session(profile_name=prior.PROFILE,region_name=prior.REGION)
    if session.client('sts').get_caller_identity()['Account']!=prior.ACCOUNT: raise ValueError('Wrong account')
    sm=session.client('sagemaker'); s3=session.client('s3')
    active=[]
    for status in ('InProgress','Stopping'):
        for page in sm.get_paginator('list_training_jobs').paginate(StatusEquals=status):
            active.extend(row['TrainingJobName'] for row in page['TrainingJobSummaries'])
    if active: raise ValueError('Do not overlap paid training jobs: '+repr(active))
    systems,order=prior.build_systems()
    order=[n for n in order if systems[n]['runtime']!='transformer-cpu']
    systems={n:systems[n] for n in order}
    assert len(systems)==17
    for spec in systems.values():
        for key in ['timeout_seconds','full_index','gate_mode','gate_band','stage_key']:
            spec.pop(key,None)
    spec=systems['tuklas-pruned']; objects={}
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=prior.BUCKET,Prefix=spec['s3_prefix']):
        for row in page.get('Contents',[]):
            rel=row['Key'][len(spec['s3_prefix']):]
            if not rel or rel.startswith('/') or '..' in Path(rel).parts: raise ValueError('Invalid package path')
            head=s3.head_object(Bucket=prior.BUCKET,Key=row['Key'])
            objects[rel]={'key':row['Key'],'bytes':row['Size'],'etag':head['ETag'].strip('"')}
            if rel=='model.safetensors':
                objects[rel]['sha256']='943414b3d9d2ef7e695df8f46d07a3b0caadfb91311673f237e74a5064fab878'
            print('PINNED pruned/'+rel,flush=True)
    if not {'model.safetensors','config.json','tokenizer.json'}<=objects.keys(): raise ValueError('Missing pruned files')
    # Pin to the intact package used by the successful AWS runtime audit.
    # Full S3 stream hashes are frozen into this run, alongside the already pinned ETag.
    head=s3.head_object(Bucket=prior.BUCKET,Key=spec['s3_prefix']+'model.safetensors')
    if head['ETag'].strip('"')!=spec['weights_etag']: raise ValueError('Pruned weights changed')
    spec['s3_objects']=objects
    sources={
        'aws_quality_metrics_repair.py':ROOT/'scripts/aws_quality_metrics_repair.py',
        'quality_metrics_complete.py':ROOT/'scripts/quality_metrics_complete.py',
        'aws_runtime_comparison_v2.py':ROOT/'scripts/aws_runtime_comparison_v2.py',
        'multiwikiqa_evaluation.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_evaluation.py',
        'multiwikiqa_ranking_v5.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py',
        'multiwikiqa_compact_retrieval.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py'}
    large={}
    for name,(source,expected) in prior.LARGE_INPUTS.items():
        key=source+name
        head=s3.head_object(Bucket=prior.BUCKET,Key=key)
        if head.get('Metadata',{}).get('sha256')!=expected: raise ValueError('Input metadata changed: '+name)
        large[name]={'key':key,'sha256':expected,'etag':head['ETag']}
        print('PINNED input/'+name+'; worker verifies full bytes',flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    job=f'pycon-quality-repair-{stamp}-{uuid4().hex[:6]}'
    prefix=f'pycon-davao-model2vec/{job}/'
    plan={'bucket':prior.BUCKET,'region':prior.REGION,'job':job,'output_prefix':prefix+'results/',
        'systems':systems,'order':order,'large_inputs':large,
        'staged_sha256':{**{n:digest(p) for n,p in sources.items()},**{n:v['sha256'] for n,v in large.items()}},
        'protocol':{'window':192,'stride':128,'chunks':40152,'parents':1294,
            'development':197,'heldout':3984,'language':'tl','ranking':'max chunk cosine per parent, ten unique parents',
            'selection':False,'latency_benchmark':False,'existing_latency_results':'unchanged',
            'metrics':['Recall@1','Recall@3','Recall@5','Recall@10','MRR@10','nDCG@10',
                       'AnswerSpanCoverage@1','AnswerSpanCoverage@3','AnswerSpanCoverage@5','AnswerSpanCoverage@10'],
            'preserve_per_question_rankings':True,'classical_models':'explicit reproductions; original weights absent',
            'tss':'official weighted inference primary; historical unweighted variant also reported'},
        'cost':{'training_usd_hour':TRAINING_USD_PER_HOUR,'runtime_seconds_cap':MAX_RUNTIME_SECONDS,
                'max_runtime_compute_usd':round(TRAINING_USD_PER_HOUR*MAX_RUNTIME_SECONDS/3600,4),
                'credits':'unverified','storage_logs_tax':'additional; not included in runtime compute cap'}}
    request={'TrainingJobName':job,'RoleArn':prior.ROLE,
        'AlgorithmSpecification':{'TrainingImage':prior.IMAGE,'TrainingInputMode':'File',
            'ContainerArguments':['python','/opt/ml/input/data/input/aws_quality_metrics_repair.py']},
        'InputDataConfig':[{'ChannelName':'input','DataSource':{'S3DataSource':{
            'S3DataType':'S3Prefix','S3Uri':f's3://{prior.BUCKET}/{prefix}input/','S3DataDistributionType':'FullyReplicated'}}}],
        'OutputDataConfig':{'S3OutputPath':f's3://{prior.BUCKET}/{prefix}native-output/'},
        'ResourceConfig':{'InstanceType':'ml.g4dn.4xlarge','InstanceCount':1,'VolumeSizeInGB':100},
        'StoppingCondition':{'MaxRuntimeInSeconds':MAX_RUNTIME_SECONDS},
        'Tags':[{'Key':'pycon-davao-model2vec','Value':'quality-metrics-repair-no-latency'}]}
    if not args.launch:
        print(json.dumps({'dry_run':True,'models':order,'protocol':plan['protocol'],'cost':plan['cost'],
                          'pruned_sha256':objects['model.safetensors']['sha256']},indent=2)); return
    run=ROOT/'runs'/job; run.mkdir(exist_ok=False)
    def save(name,value):
        with (run/name).open('x') as f: json.dump(value,f,indent=2,sort_keys=True,default=str)
    save('plan.json',plan); save('create-training-job-request.json',request)
    sources['plan.json']=run/'plan.json'
    archive=run/'sources'; archive.mkdir()
    for name,path in sources.items():
        data=path.read_bytes()
        if name!='plan.json':
            with (archive/name).open('xb') as f:f.write(data)
        s3.put_object(Bucket=prior.BUCKET,Key=prefix+'input/'+name,Body=data,
            IfNoneMatch='*',ServerSideEncryption='AES256',Metadata={'sha256':sha256(data).hexdigest()})
    for name,spec in large.items():
        # New UUID prefix; server-side copy avoids slow local round trips.
        dest=prefix+'input/'+name
        try:
            s3.head_object(Bucket=prior.BUCKET,Key=dest)
        except s3.exceptions.ClientError as error:
            if error.response['Error']['Code'] not in ('404','NoSuchKey'):raise
        else:raise ValueError('Refuse overwrite: '+dest)
        s3.copy_object(Bucket=prior.BUCKET,Key=dest,CopySource={'Bucket':prior.BUCKET,'Key':spec['key']},
            CopySourceIfMatch=spec['etag'],ServerSideEncryption='AES256',MetadataDirective='REPLACE',
            Metadata={'sha256':spec['sha256']})
    # Recheck immediately before the sole paid side effect.
    for status in ('InProgress','Stopping'):
        if sm.list_training_jobs(StatusEquals=status,MaxResults=1)['TrainingJobSummaries']:
            raise RuntimeError('A training job started during staging; new job not launched')
    save('create-training-job-response.json',sm.create_training_job(**request))
    print(json.dumps({'job':job,'run':str(run),'results':f's3://{prior.BUCKET}/{prefix}results/','cost':plan['cost']},indent=2))


if __name__=='__main__':main()
