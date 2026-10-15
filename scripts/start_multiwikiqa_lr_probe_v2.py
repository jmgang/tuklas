"""Stage and launch the exact-schedule LR probe with immutable source hashes."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4
import boto3

ROOT=Path(__file__).resolve().parents[1]
PRIOR='pycon-multiwikiqa-ranking-repair-20261004-153209-f6e9d7'
FAILED_PROBE='pycon-multiwikiqa-lr-probe-20261004-161916-7a2967'
BUCKET='sagemaker-ap-southeast-1-<aws-account-id>'


def digest(path):
    h=sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def save(path,obj):
    with path.open('x') as target:
        json.dump(obj,target,sort_keys=True,indent=2,default=str)
        target.write('\n')


def main():
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    if session.client('sts').get_caller_identity()['Account']!='<aws-account-id>':
        raise ValueError('Wrong AWS account')
    s3,sm=session.client('s3'),session.client('sagemaker')
    if sm.describe_training_job(TrainingJobName=PRIOR)['TrainingJobStatus']!='Completed':
        raise ValueError('Frozen parent job not completed')
    if sm.describe_training_job(TrainingJobName=FAILED_PROBE)['TrainingJobStatus']!='Stopped':
        raise ValueError('Invalid prior probe not established stopped')
    for prefix in ('pycon-multiwikiqa','pycon-sealion'):
        for page in sm.get_paginator('list_training_jobs').paginate(NameContains=prefix,StatusEquals='InProgress'):
            if page['TrainingJobSummaries']:
                raise ValueError('Another experimental job is active')
    prior=ROOT/'runs'/PRIOR
    old=json.loads((prior/'mixed-plan.json').read_text())
    if old['ranking_preflight']!={'train_pairs':1734,'validation_questions':338,'heldout_questions':7922,
        'skipped_training_questions':6,'batch_sizes':[2,26,78,141,178,202,215,219,221,222]}:
        raise ValueError('Frozen train/validation layout changed')
    name='pycon-multiwikiqa-lr-probe-v2-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid4().hex[:6]
    directory=ROOT/'runs'/name
    directory.mkdir(exist_ok=False)
    prefix='pycon-davao-model2vec/'+name
    archive=ROOT/'runs/aws-demo-source-index-20261004T132400Z/exact/code/pycon-sealion-repair-20261004-084939-91585a'
    files={
        'aws_multiwikiqa_lr_probe_v2.py':ROOT/'scripts/aws_multiwikiqa_lr_probe_v2.py',
        'aws_sealion_train_repair_v3.py':archive/'repair/aws_sealion_train_repair_v3.py',
        'sealion_runtime.py':archive/'repair/sealion_runtime.py',
        'sealion_source_integrity.py':archive/'repair/sealion_source_integrity.py',
        'sealion_training.py':archive/'input/sealion_training.py',
        'multiwikiqa_evaluation.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_evaluation.py',
        'multiwikiqa_ranking.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_ranking_v2.py',
    }
    for path in files.values():
        compile(path.read_text(),str(path),'exec')
    save(directory/'settings.json',json.loads((prior/'settings.json').read_text()))
    files['settings.json']=directory/'settings.json'
    plan={'bucket':BUCKET,'frozen_student':old['frozen_students']['approved_articles'],
          'annotations':old['annotations'],'evaluation_layout':old['evaluation_layout'],
          'ranking':old['ranking'],
          'learning_rates':[['control_0p05',0.05],['lower_0p005',0.005],['lower_0p0005',0.0005]],
          'experiment':'Only learning rate varies; preserve original three-epoch scheduler and stop after exact first epoch. No held-out scoring.',
          'source_files':{n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()}}
    save(directory/'probe-plan.json',plan)
    files['probe-plan.json']=directory/'probe-plan.json'
    for filename,path in files.items():
        with path.open('rb') as source:
            s3.put_object(Bucket=BUCKET,Key=prefix+'/input/'+filename,Body=source,
                          IfNoneMatch='*',Metadata={'sha256':digest(path)})
        head=s3.head_object(Bucket=BUCKET,Key=prefix+'/input/'+filename)
        if head['ContentLength']!=path.stat().st_size or head['Metadata'].get('sha256')!=digest(path):
            raise ValueError('Staged source checksum mismatch')
    request=json.loads((prior/'create-training-job-request.json').read_text())
    request['TrainingJobName']=name
    request['AlgorithmSpecification']['ContainerArguments']=['python /opt/ml/input/data/input/aws_multiwikiqa_lr_probe_v2.py']
    request['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri']='s3://'+BUCKET+'/'+prefix+'/input/'
    request['OutputDataConfig']['S3OutputPath']='s3://'+BUCKET+'/'+prefix+'/native-output/'
    request['StoppingCondition']['MaxRuntimeInSeconds']=3600
    request['Environment']['SEALION_OUTPUT_PREFIX']=prefix+'/results'
    request['Environment'].pop('SEALION_RECOVERY_MANIFEST',None)
    request['Tags']=[{'Key':'Project','Value':'pycon-davao-model2vec-demo'},
                     {'Key':'Purpose','Value':'exact-schedule-development-lr-probe'},
                     {'Key':'DemoSourceManifest','Value':'s3://'+BUCKET+'/'+prefix+'/input/probe-plan.json'}]
    save(directory/'create-training-job-request.json',request)
    save(directory/'resource-record.json',{
        'job_name':name,'region':'ap-southeast-1','profile':'admin-general',
        's3_results':'s3://'+BUCKET+'/'+prefix+'/results/',
        'max_runtime_seconds':3600,'instance':'ml.g4dn.2xlarge','rate_usd_per_hour':1.315,
        'max_runtime_compute_estimate_usd':1.315,'credits_verified':False,
        'no_endpoint':True,'managed_release_on_terminal':True,'heldout_scored':False,
        'invalid_prior_probe':FAILED_PROBE,
        'source_files':{n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()}})
    response=sm.create_training_job(**request)
    save(directory/'launch-response.json',response)
    state=sm.describe_training_job(TrainingJobName=name)
    save(directory/'initial-job-state.json',state)
    print(json.dumps({'job_name':name,'status':state['TrainingJobStatus'],
                      's3_results':'s3://'+BUCKET+'/'+prefix+'/results/'}))


if __name__=='__main__':
    main()
