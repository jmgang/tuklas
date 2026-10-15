"""Stage first, then explicitly launch the authorized mixture/ranking experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from uuid import uuid4
import boto3

ROOT=Path(__file__).resolve().parents[1]
BUCKET='sagemaker-ap-southeast-1-<aws-account-id>'
TRAIN='pycon-sealion-repair-20261004-084939-91585a'
EVAL='pycon-multiwikiqa-eval-20261004-131724-611940'
SOURCE='pycon-davao-model2vec/'+TRAIN+'/results/'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,default=str);f.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch-authorized',type=Path)
    args=parser.parse_args()
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    if session.client('sts').get_caller_identity()['Account']!='<aws-account-id>':
        raise ValueError('Wrong account')
    sm,s3=session.client('sagemaker'),session.client('s3')
    for name in ('pycon-multiwikiqa-mixed','pycon-sealion','pycon-multiwikiqa-eval'):
        for page in sm.get_paginator('list_training_jobs').paginate(NameContains=name,StatusEquals='InProgress'):
            if page['TrainingJobSummaries']:raise ValueError('Active conflicting job; refusing duplicate')
    if args.launch_authorized:
        directory=args.launch_authorized.resolve()
        if not directory.is_relative_to(ROOT/'runs'):raise ValueError('Outside project')
        record=json.loads((directory/'resource-record.json').read_text())
        request=json.loads((directory/'create-training-job-request.json').read_text())
        if digest(directory/'create-training-job-request.json')!=record['request_sha256']:
            raise ValueError('Prepared request changed')
        if (request['StoppingCondition']['MaxRuntimeInSeconds']!=7200 or
                request['ResourceConfig']['InstanceType']!='ml.g4dn.2xlarge'):
            raise ValueError('Cost/compute protocol changed')
        for name,spec in record['source_files'].items():
            head=s3.head_object(Bucket=BUCKET,Key=record['input_prefix']+'/'+name)
            if head['ContentLength']!=spec['bytes'] or head['Metadata'].get('sha256')!=spec['sha256']:
                raise ValueError('Staged input changed')
        response=sm.create_training_job(**request)
        save(directory/'launch-response.json',response)
        state=sm.describe_training_job(TrainingJobName=record['job_name'])
        save(directory/'initial-job-state.json',state)
        print(json.dumps({'job_name':record['job_name'],'status':state['TrainingJobStatus'],
                         'run_directory':str(directory),'s3_results':record['s3_results']}));return
    for name in (TRAIN,EVAL):
        if sm.describe_training_job(TrainingJobName=name)['TrainingJobStatus']!='Completed':
            raise ValueError('Prerequisite not complete')
    old=ROOT/'runs'/TRAIN/'verification-20261004T113545Z/completion.json'
    if digest(old)!='a0d512ebaa2d181a94cd60c3d2f7eecfd2eb9452555361b49655e6a1de39359b':
        raise ValueError('Verified training record changed')
    completed=json.loads(old.read_text())
    artifacts=completed['artifacts']
    def spec(relative):return {**artifacts[relative],'key':SOURCE+relative}
    arms={}
    for arm,windows in [('external_wikipedia',176178),('approved_articles',6546)]:
        names=sorted(n for n in artifacts if n.startswith('features/'+arm+'/') and n.endswith('.json'))
        arms[arm]={'windows':windows,'layout':spec('layouts/'+arm+'-training-layout.jsonl'),
                  'pairs':[{'text':spec(n),'vectors':spec(n[:-5]+'.npy')} for n in names]}
        starts=[int(n.rsplit('-',1)[-1][:-5]) for n in names]
        if starts!=list(range(0,windows,1024)):
            raise ValueError('Frozen targets incomplete/noncontiguous')
    students={arm:{n.rsplit('/',1)[-1]:spec(n) for n in artifacts if n.startswith('models/'+arm+'/')}
              for arm in ('approved_articles','external_wikipedia')}
    initial={n.rsplit('/',1)[-1]:spec(n) for n in artifacts if n.startswith('models/initialization/')}
    recovery={'bucket':BUCKET,'source_job':'pycon-sealion-20261004-073934-2eccba','files':initial}
    ep='pycon-davao-model2vec/'+EVAL
    complete=json.loads(s3.get_object(Bucket=BUCKET,Key=ep+'/results/reports/completion.json')['Body'].read())
    layout_spec={**complete['artifacts']['shared-layout.json'],'key':ep+'/results/shared-layout.json'}
    raw=s3.get_object(Bucket=BUCKET,Key=layout_spec['key'])['Body'].read()
    if len(raw)!=layout_spec['bytes'] or sha256(raw).hexdigest()!=layout_spec['sha256']:
        raise ValueError('Shared layout checksum changed')
    layout=json.loads(raw)
    oldsettings=json.loads((ROOT/'runs'/EVAL/'settings.json').read_text())
    annotations={**oldsettings['data_files']['annotations.all.jsonl'],'key':ep+'/input/annotations.all.jsonl'}
    qa_raw=s3.get_object(Bucket=BUCKET,Key=annotations['key'])['Body'].read()
    if len(qa_raw)!=annotations['bytes'] or sha256(qa_raw).hexdigest()!=annotations['sha256']:
        raise ValueError('QA checksum changed')
    queries=[json.loads(line) for line in qa_raw.decode().splitlines()]
    sys.path.insert(0,str(ROOT/'src'))
    from pycon_model2vec_demo.multiwikiqa_ranking import prepare,batches
    train,validation,skipped=prepare(queries,layout['chunks'])
    if len([q for q in queries if q['partition']=='evaluation'])!=7922:
        raise ValueError('Heldout cohort changed')
    ranking={'method':'Potion-style MNRL plus Matryoshka; approved bilingual development pairs',
        'epochs':3,'learning_rate':0.05,'batch_size':2048,'warmup_ratio':0.1,'seed':42,
        'matryoshka_dimensions':[32,64,128,256,512,1024],
        'precision':'float32; no quantization','selection':'development component validation nDCG@10',
        'exact_potion_reproduction':False,'differences_from_potion':'teacher, bilingual data, full 1024 output, F32, smaller effective unique-component batches; no extra datasets'}
    plan={'bucket':BUCKET,'arms':arms,'frozen_students':students,'evaluation_layout':layout_spec,
        'annotations':annotations,'ranking':ranking,'ranking_preflight':{
            'train_pairs':len(train),'validation_questions':len(validation),'heldout_questions':7922,
            'skipped_training_questions':len(skipped),'batch_sizes':sorted(set(map(len,batches(train,2048))))},
        'mixture':'existing source union only; 175658 external plus 6261 approved source rows',
        'QA_training':'development only; evaluation QA explicitly forbidden'}
    plan['frozen_baseline_reports']={name:{**complete['artifacts']['reports/'+name+'.json'],
        'key':ep+'/results/reports/'+name+'.json'} for name in
        ('teacher','approved_articles','external_wikipedia','potion_retrieval')}
    allspecs=list(initial.values())+list(students['approved_articles'].values())+list(students['external_wikipedia'].values())
    allspecs += [a['layout'] for a in arms.values()]+[x for a in arms.values() for p in a['pairs'] for x in p.values()]
    def check(x):
        head=s3.head_object(Bucket=BUCKET,Key=x['key'])
        if head['ContentLength']!=x['bytes'] or head['Metadata'].get('sha256')!=x['sha256']:
            raise ValueError('Previously verified input metadata changed: '+x['key'])
    with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(check,allspecs))
    name='pycon-multiwikiqa-mixed-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid4().hex[:6]
    directory=ROOT/'runs'/name;directory.mkdir(exist_ok=False)
    prefix='pycon-davao-model2vec/'+name
    archive=ROOT/'runs/aws-demo-source-index-20261004T132400Z/exact/code'/TRAIN
    files={'aws_multiwikiqa_mixed_student.py':ROOT/'scripts/aws_multiwikiqa_mixed_student.py',
           'aws_sealion_train_repair_v3.py':archive/'repair/aws_sealion_train_repair_v3.py',
           'sealion_runtime.py':archive/'repair/sealion_runtime.py',
           'sealion_source_integrity.py':archive/'repair/sealion_source_integrity.py',
           'sealion_training.py':archive/'input/sealion_training.py',
           'multiwikiqa_evaluation.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_evaluation.py',
           'multiwikiqa_ranking.py':ROOT/'src/pycon_model2vec_demo/multiwikiqa_ranking.py'}
    for path in files.values():compile(path.read_text(),str(path),'exec')
    config=json.loads((ROOT/'configs/sealion-two-students-v1.json').read_text())
    config['arms']['approved_plus_external']={'interpretation':'simple approved+external text union; same initial model; unequal arm sizes remain disclosed'}
    config['authorization']['additional_stage']='User explicitly requested approved+external student and approved development-only ranking training for all three students.'
    save(directory/'settings.json',config);files['settings.json']=directory/'settings.json'
    plan['source_files']={n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()}
    save(directory/'mixed-plan.json',plan);files['mixed-plan.json']=directory/'mixed-plan.json'
    save(directory/'initialization-recovery.json',recovery);files['initialization-recovery.json']=directory/'initialization-recovery.json'
    for filename,path in files.items():
        with path.open('rb') as f:s3.put_object(Bucket=BUCKET,Key=prefix+'/input/'+filename,
            Body=f,IfNoneMatch='*',Metadata={'sha256':digest(path)})
    oldrequest=json.loads((ROOT/'runs'/TRAIN/'create-training-job-request.json').read_text())
    request={k:oldrequest[k] for k in ('AlgorithmSpecification','RoleArn','EnableNetworkIsolation')}
    request['AlgorithmSpecification']={**request['AlgorithmSpecification'],
        'ContainerArguments':['python /opt/ml/input/data/input/aws_multiwikiqa_mixed_student.py']}
    request.update({'TrainingJobName':name,'ResourceConfig':{'InstanceType':'ml.g4dn.2xlarge','InstanceCount':1,'VolumeSizeInGB':60},
        'StoppingCondition':{'MaxRuntimeInSeconds':7200},
        'InputDataConfig':[{'ChannelName':'input','InputMode':'File','DataSource':{'S3DataSource':{
            'S3DataType':'S3Prefix','S3Uri':'s3://'+BUCKET+'/'+prefix+'/input/','S3DataDistributionType':'FullyReplicated'}}}],
        'OutputDataConfig':{'S3OutputPath':'s3://'+BUCKET+'/'+prefix+'/native-output/'},
        'Environment':{'SEALION_OUTPUT_BUCKET':BUCKET,'SEALION_OUTPUT_PREFIX':prefix+'/results',
            'SEALION_RECOVERY_MANIFEST':'/opt/ml/input/data/input/initialization-recovery.json',
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','HF_HUB_DISABLE_XET':'1',
            'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
        'Tags':[{'Key':'Project','Value':'pycon-davao-model2vec-demo'},
            {'Key':'Purpose','Value':'mixed-tokenlearn-plus-three-ranking-students'},
            {'Key':'DemoSourceManifest','Value':'s3://'+BUCKET+'/'+prefix+'/input/mixed-plan.json'}]})
    save(directory/'create-training-job-request.json',request)
    record={'job_name':name,'profile':'admin-general','region':'ap-southeast-1','run_directory':str(directory),
        'input_prefix':prefix+'/input','s3_results':'s3://'+BUCKET+'/'+prefix+'/results/',
        'rate_usd_per_hour':1.315,'max_runtime_seconds':7200,'max_training_compute_estimate_usd':2.63,
        'credits_verified':False,'no_endpoint':True,'managed_release_on_terminal':True,
        'source_files':{n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()},
        'request_sha256':digest(directory/'create-training-job-request.json'),'status':'STAGED_NOT_LAUNCHED',
        'ranking_preflight':plan['ranking_preflight']}
    save(directory/'resource-record.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
