"""Launch the explicitly authorized one-off quality evaluation, no endpoints."""
from datetime import datetime,timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
from uuid import uuid4
import boto3

ROOT=Path(__file__).resolve().parents[1]
BUCKET='sagemaker-ap-southeast-1-<aws-account-id>'
SOURCE='pycon-davao-model2vec/pycon-sealion-repair-20261004-084939-91585a/results/'


def hash_file(path):
    h=sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,default=str)
        f.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch-authorized',action='store_true')
    args=parser.parse_args()
    if not args.launch_authorized:
        raise RuntimeError('Paid launch requires user authorization')
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    if session.client('sts').get_caller_identity()['Account']!='<aws-account-id>':
        raise ValueError('Wrong account')
    sm,s3=session.client('sagemaker'),session.client('s3')
    if sm.list_training_jobs(NameContains='pycon-multiwikiqa-eval',StatusEquals='InProgress')['TrainingJobSummaries']:
        raise RuntimeError('An evaluation already runs; refusing duplicate')
    if sm.describe_training_job(TrainingJobName='pycon-sealion-repair-20261004-084939-91585a')['TrainingJobStatus']!='Completed':
        raise ValueError('Training not complete')
    prep=ROOT/'runs/multiwikiqa-preparation-20261004-v1'
    manifest=json.loads((prep/'manifest.json').read_text())
    frozen=ROOT/'runs/pycon-sealion-repair-20261004-084939-91585a/verification-20261004T113545Z/completion.json'
    if hash_file(frozen)!='a0d512ebaa2d181a94cd60c3d2f7eecfd2eb9452555361b49655e6a1de39359b':
        raise ValueError('Training completion record changed')
    complete=json.loads(frozen.read_text())
    name='pycon-multiwikiqa-eval-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid4().hex[:6]
    prefix='pycon-davao-model2vec/'+name
    directory=ROOT/'runs'/name
    directory.mkdir(exist_ok=False)
    static={}
    for arm in ('approved_articles','external_wikipedia'):
        static[arm]={}
        for path,spec in complete['artifacts'].items():
            if path.startswith('models/'+arm+'/'):
                key=SOURCE+path
                head=s3.head_object(Bucket=BUCKET,Key=key)
                if head['ContentLength']!=spec['bytes'] or head['Metadata'].get('sha256')!=spec['sha256']:
                    raise ValueError('Existing student metadata changed')
                static[arm][path.rsplit('/',1)[-1]]={**spec,'key':key}
    potion=ROOT/'models/potion-retrieval-6fc8051fab2a'
    static['potion_retrieval']={}
    for path in sorted(potion.iterdir()):
        if path.is_file():
            spec={'bytes':path.stat().st_size,'sha256':hash_file(path),
                  'source':'pinned-huggingface'}
            static['potion_retrieval'][path.name]=spec
    settings={'protocol':'multiwikiqa-as-released-quality-v1','dataset':manifest['dataset'],
        'dataset_revision':manifest['revision'],'questions':10000,'parents':1294,
        'teacher':{'model_id':'aisingapore/SEA-LION-E5-Embedding-600M',
            'revision':'6a97cedd22714f3d4eb7c6fd1a452bf310899683',
            'query_prompt':'Instruct: Given a passage that is guaranteed to contain the answer, retrieve relevant passages that answer the query.\nQuery: ',
            'document_prompt':'','dimension':1024,'precision':'float32'},
        'static_models':static,'static_query_document_prompts':'empty, matching static inference usage',
        'potion_revision':'6fc8051fab2a1e0ee76689cf08c853792ac285e7',
        'data_files':{},'cutoffs':[1,3,5,10],'parent_reduction':'max chunk cosine',
        'semantic_review':'pending; as-released evaluation',
        'empty_answers':'include source-parent retrieval; exclude only undefined span-coverage denominator',
        'no_training':True,'no_latency_benchmark':True,'no_runtime_memory_benchmark':True,
        'paired_article_cluster_bootstrap':{'samples':2000,'seed':42},
        'authorization':'User explicitly answered Yes—run the AWS evaluation, October 4, 2026; one new job with two-hour runtime cap.'}
    files={}
    for filename in ('parents.jsonl','annotations.all.jsonl','chunks.candidate.jsonl'):
        path=prep/filename
        checksum=hash_file(path)
        if checksum!=manifest['artifacts_sha256'][filename]:
            raise ValueError('Prepared data changed: '+filename)
        settings['data_files'][filename]={'bytes':path.stat().st_size,'sha256':checksum}
        files[filename]=path
    save(directory/'settings.json',settings)
    files.update({'settings.json':directory/'settings.json',
        'aws_multiwikiqa_evaluate_v2.py':ROOT/'scripts/aws_multiwikiqa_evaluate_v2.py',
        'multiwikiqa_evaluation.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_evaluation.py'})
    save(directory/'staging-plan.json',{'settings':settings,'source_files':{
        n:{'sha256':hash_file(p),'bytes':p.stat().st_size,'local_path':str(p)} for n,p in files.items()}})
    for filename,path in files.items():
        with path.open('rb') as f:
            s3.put_object(Bucket=BUCKET,Key=prefix+'/input/'+filename,Body=f,
                ContentLength=path.stat().st_size,IfNoneMatch='*',Metadata={'sha256':hash_file(path)})
        print('STAGED',filename,flush=True)
    original=json.loads((ROOT/'runs/pycon-sealion-repair-20261004-084939-91585a/create-training-job-request.json').read_text())
    request={k:original[k] for k in ('AlgorithmSpecification','RoleArn','ResourceConfig','EnableNetworkIsolation')}
    request['AlgorithmSpecification']={**request['AlgorithmSpecification'],
        'ContainerArguments':['python /opt/ml/input/data/input/aws_multiwikiqa_evaluate_v2.py']}
    request.update({'TrainingJobName':name,'StoppingCondition':{'MaxRuntimeInSeconds':7200},
        'ResourceConfig':{**request['ResourceConfig'],'VolumeSizeInGB':40},
        'InputDataConfig':[{'ChannelName':'input','InputMode':'File','DataSource':{'S3DataSource':{
            'S3DataType':'S3Prefix','S3Uri':'s3://'+BUCKET+'/'+prefix+'/input/','S3DataDistributionType':'FullyReplicated'}}}],
        'OutputDataConfig':{'S3OutputPath':'s3://'+BUCKET+'/'+prefix+'/native-output/'},
        'Environment':{'EVAL_BUCKET':BUCKET,'EVAL_PREFIX':prefix+'/results','PYTHONDONTWRITEBYTECODE':'1',
            'HF_HUB_DISABLE_XET':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
        'Tags':[{'Key':'Project','Value':'pycon-davao-model2vec-demo'},{'Key':'Purpose','Value':'10000-question-quality-evaluation'}]})
    save(directory/'create-training-job-request.json',request)
    response=sm.create_training_job(**request)
    save(directory/'launch-response.json',response)
    state=sm.describe_training_job(TrainingJobName=name)
    save(directory/'initial-job-state.json',state)
    record={'job_name':name,'profile':'admin-general','region':'ap-southeast-1',
        's3_results':'s3://'+BUCKET+'/'+prefix+'/results/','run_directory':str(directory),
        'instance_type':'ml.g4dn.2xlarge','max_runtime_seconds':7200,'rate_usd_per_hour':1.315,
        'credits_verified':False,'status':state['TrainingJobStatus'],'purpose':'retrieval evaluation, NOT training'}
    save(directory/'resource-record.json',record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
