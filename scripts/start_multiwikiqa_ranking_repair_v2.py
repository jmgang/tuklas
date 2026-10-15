"""Authorized bounded ranking-only repair; no repeated TokenLearn training."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4
import boto3

ROOT=Path(__file__).resolve().parents[1]
OLD='pycon-multiwikiqa-mixed-20261004-143440-0eb179'
BUCKET='sagemaker-ap-southeast-1-<aws-account-id>'


def digest(path):return sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,default=str);f.write('\n')


def main():
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    if session.client('sts').get_caller_identity()['Account']!='<aws-account-id>':raise ValueError('Wrong account')
    s3,sm=session.client('s3'),session.client('sagemaker')
    if sm.describe_training_job(TrainingJobName=OLD)['TrainingJobStatus']!='Failed':
        raise ValueError('Original worker not established terminal Failed')
    for prefix in ('pycon-multiwikiqa-mixed','pycon-multiwikiqa-ranking-repair'):
        for page in sm.get_paginator('list_training_jobs').paginate(NameContains=prefix,StatusEquals='InProgress'):
            if page['TrainingJobSummaries']:raise ValueError('Active worker; refusing duplicate')
    olddir=ROOT/'runs'/OLD
    oldrecord=json.loads((olddir/'resource-record.json').read_text())
    oldplan=olddir/'mixed-plan.json'
    if digest(oldplan)!=oldrecord['source_files']['mixed-plan.json']['sha256']:
        raise ValueError('Original frozen plan changed')
    plan=json.loads(oldplan.read_text())
    source='pycon-davao-model2vec/'+OLD+'/results/'
    def spec(relative):
        key=source+relative;head=s3.head_object(Bucket=BUCKET,Key=key)
        checksum=head['Metadata'].get('sha256')
        if not checksum or len(checksum)!=64:raise ValueError('Missing worker checksum')
        return {'key':key,'bytes':head['ContentLength'],'sha256':checksum}
    plan['recovered_training_report']=spec('reports/approved_plus_external-training-report.json')
    plan['recovered_quality']={name:spec('reports/'+name+'-quality.json') for name in ('initialization','approved_plus_external')}
    for name,item in plan['recovered_quality'].items():
        raw=s3.get_object(Bucket=BUCKET,Key=item['key'])['Body'].read()
        if sha256(raw).hexdigest()!=item['sha256'] or len(raw)!=item['bytes']:raise ValueError('Quality bytes changed')
        report=json.loads(raw)
        if report['shared_layout_sha256']!=plan['evaluation_layout']['sha256'] or report['question_count']!=10000:
            raise ValueError('Quality provenance changed')
        item['model_weights_sha256']=report['model_weights_sha256']
    expected={'README.md','config.json','modules.json','model.safetensors','tokenizer.json'}
    plan['recovered_mixture']={name:spec('models/approved_plus_external/'+name) for name in sorted(expected)}
    if plan['recovered_mixture']['model.safetensors']['sha256']!=plan['recovered_quality']['approved_plus_external']['model_weights_sha256']:
        raise ValueError('Saved mixture does not match completed quality report')
    if plan['ranking_preflight']!={'train_pairs':1734,'validation_questions':338,'heldout_questions':7922,
            'skipped_training_questions':6,'batch_sizes':[2,26,78,141,178,202,215,219,221,222]}:
        raise ValueError('Ranking recipe/split changed')
    plan['repair']='Device transfer in export-parity forward only; original recipe/cohort/weights unchanged. Completed mixed student reused, not retrained.'
    name='pycon-multiwikiqa-ranking-repair-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid4().hex[:6]
    directory=ROOT/'runs'/name;directory.mkdir(exist_ok=False)
    prefix='pycon-davao-model2vec/'+name
    archive=ROOT/'runs/aws-demo-source-index-20261004T132400Z/exact/code/pycon-sealion-repair-20261004-084939-91585a'
    files={'aws_multiwikiqa_ranking_repair_v2.py':ROOT/'scripts/aws_multiwikiqa_ranking_repair_v2.py',
        'aws_sealion_train_repair_v3.py':archive/'repair/aws_sealion_train_repair_v3.py',
        'sealion_runtime.py':archive/'repair/sealion_runtime.py',
        'sealion_source_integrity.py':archive/'repair/sealion_source_integrity.py',
        'sealion_training.py':archive/'input/sealion_training.py',
        'multiwikiqa_evaluation.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_evaluation.py',
        'multiwikiqa_ranking.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_ranking_v2.py'}
    for p in files.values():compile(p.read_text(),str(p),'exec')
    save(directory/'settings.json',json.loads((olddir/'settings.json').read_text()))
    files['settings.json']=directory/'settings.json'
    plan['source_files']={n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()}
    save(directory/'mixed-plan.json',plan);files['mixed-plan.json']=directory/'mixed-plan.json'
    for filename,path in files.items():
        with path.open('rb') as f:s3.put_object(Bucket=BUCKET,Key=prefix+'/input/'+filename,Body=f,
            IfNoneMatch='*',Metadata={'sha256':digest(path)})
        head=s3.head_object(Bucket=BUCKET,Key=prefix+'/input/'+filename)
        if head['ContentLength']!=path.stat().st_size or head['Metadata'].get('sha256')!=digest(path):
            raise ValueError('Staging checksum mismatch')
    request=json.loads((olddir/'create-training-job-request.json').read_text())
    request['TrainingJobName']=name
    request['AlgorithmSpecification']['ContainerArguments']=['python /opt/ml/input/data/input/aws_multiwikiqa_ranking_repair_v2.py']
    request['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri']='s3://'+BUCKET+'/'+prefix+'/input/'
    request['OutputDataConfig']['S3OutputPath']='s3://'+BUCKET+'/'+prefix+'/native-output/'
    request['StoppingCondition']['MaxRuntimeInSeconds']=1800
    request['Environment']['SEALION_OUTPUT_PREFIX']=prefix+'/results'
    request['Environment'].pop('SEALION_RECOVERY_MANIFEST',None)
    request['Tags']=[{'Key':'Project','Value':'pycon-davao-model2vec-demo'},
        {'Key':'Purpose','Value':'ranking-only-device-repair'},
        {'Key':'DemoSourceManifest','Value':'s3://'+BUCKET+'/'+prefix+'/input/mixed-plan.json'}]
    save(directory/'create-training-job-request.json',request)
    record={'job_name':name,'profile':'admin-general','region':'ap-southeast-1','run_directory':str(directory),
        's3_results':'s3://'+BUCKET+'/'+prefix+'/results/','input_prefix':prefix+'/input',
        'rate_usd_per_hour':1.315,'max_runtime_seconds':1800,'max_training_compute_estimate_usd':0.6575,
        'credits_verified':False,'no_endpoint':True,'managed_release_on_terminal':True,
        'source_files':{n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()},
        'recovered_completed_student':OLD,'no_repeated_tokenlearn_training':True,
        'authorization':'User responded to bounded repair launch question: you need to keep fixing and stop asking questions.',
        'status':'STAGED'}
    save(directory/'resource-record.json',record)
    response=sm.create_training_job(**request);save(directory/'launch-response.json',response)
    state=sm.describe_training_job(TrainingJobName=name);save(directory/'initial-job-state.json',state)
    print(json.dumps({**record,'status':state['TrainingJobStatus']},indent=2))


if __name__=='__main__':main()
