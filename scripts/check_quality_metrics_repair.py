#!/usr/bin/env python3
"""Read-only AWS status; --verify saves independently replayed results in a NEW folder."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/pycon_model2vec_demo')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--verify',action='store_true');parser.add_argument('--resume-directory',type=Path)
    args=parser.parse_args();run=args.run.resolve();plan=json.loads((run/'plan.json').read_text())
    session=boto3.Session(profile_name='admin-general',region_name='ap-southeast-1')
    sm=session.client('sagemaker');s3=session.client('s3',config=Config(read_timeout=300,connect_timeout=30,retries={'mode':'standard','max_attempts':5}))
    state=sm.describe_training_job(TrainingJobName=plan['job'])
    summary={k:state.get(k) for k in ['TrainingJobName','TrainingJobStatus','SecondaryStatus','FailureReason','BillableTimeInSeconds','TrainingStartTime','TrainingEndTime']}
    if state.get('BillableTimeInSeconds') is not None:summary['estimated_compute_usd']=state['BillableTimeInSeconds']*2.106/3600
    entries=[]
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=plan['bucket'],Prefix=plan['output_prefix']+'events/'):
        entries+=page.get('Contents',[])
    if entries:
        key=max(entries,key=lambda x:x['Key'])['Key']
        summary['latest_event']=json.loads(s3.get_object(Bucket=plan['bucket'],Key=key)['Body'].read())
    print(json.dumps(summary,indent=2,default=str))
    if not args.verify:return
    if state['TrainingJobStatus']!='Completed':raise SystemExit('Verification requires Completed state')
    from quality_metrics_complete import aggregate,METRICS
    from multiwikiqa_evaluation import score
    if args.resume_directory:
        target=args.resume_directory.resolve()
        if target.parent!=run or not target.name.startswith('verified-') or not target.is_dir():
            raise ValueError('Resume directory must be an existing verification directory within this run')
    else:
        target=run/('verified-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));target.mkdir(exist_ok=False)
    def get(rel):
        path=target/rel
        if path.is_file():
            data=path.read_bytes()
            head=s3.head_object(Bucket=plan['bucket'],Key=plan['output_prefix']+rel)
            expected=head.get('Metadata',{}).get('sha256')
            if not expected or sha256(data).hexdigest()!=expected:
                raise ValueError('Cached S3 checksum mismatch: '+rel)
            return data
        for attempt in range(5):
            try:
                response=s3.get_object(Bucket=plan['bucket'],Key=plan['output_prefix']+rel)
                data=response['Body'].read();expected=response.get('Metadata',{}).get('sha256')
                if not expected or sha256(data).hexdigest()!=expected:raise ValueError('S3 checksum mismatch: '+rel)
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:f.write(data)
                return data
            except (BotoCoreError,ClientError,TimeoutError):
                if attempt==4:raise
    names=['complete-report.json','evidence/corpus.json','evidence/development.json','evidence/heldout.json']
    names += ['rankings/'+name+'-'+cohort+'.jsonl' for name in plan['order']
              for cohort in ('development','heldout')]
    names += ['rankings/tss-m2v-bge-m3-1024d-legacy-unweighted-'+cohort+'.jsonl'
              for cohort in ('development','heldout')]
    # S3-to-local throughput is connection limited; parallel independent GETs.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get,names))
    report=json.loads(get('complete-report.json'))
    if report['status']!='complete' or set(report['results'])!=set(plan['order']) or report['failures']:
        raise ValueError('Model roster incomplete')
    corpus=json.loads(get('evidence/corpus.json'));cohorts={n:json.loads(get('evidence/'+n+'.json')) for n in ('development','heldout')}
    from aws_runtime_comparison_v2 import build_index
    from multiwikiqa_ranking_v5 import prepare
    def frozen_input(name):
        item=plan['large_inputs'][name]
        local=(ROOT/'runs/pycon-compact-sota-20261005-055331-d48482/verification-20261005T090654Z/evaluation'/name)
        if local.is_file():
            data=local.read_bytes()
        else:
            data=s3.get_object(Bucket=plan['bucket'],Key=item['key'])['Body'].read()
        if sha256(data).hexdigest()!=item['sha256']:raise ValueError('Frozen input changed')
        return data
    layout=json.loads(frozen_input('shared-layout.json'))
    records,owners=build_index(layout)
    if corpus!={'records':records,'owners':owners,'parents':layout['parents']}:raise ValueError('Rebuilt index differs')
    annotations=[json.loads(line) for line in frozen_input('annotations.all.jsonl').splitlines() if line.strip()]
    tl=[q for q in annotations if q['language']=='tl'];_,dev,_=prepare(tl,layout['chunks'])
    if cohorts!={'development':dev,'heldout':[q for q in tl if q['partition']=='evaluation']}:raise ValueError('Independent cohorts differ')
    assert len(corpus['records'])==40152 and len(corpus['parents'])==1294
    assert len(cohorts['development'])==197 and len(cohorts['heldout'])==3984
    verified=0
    for name,result in report['results'].items():
        variants={name:result}
        if 'legacy_unweighted' in result:variants[name+'-legacy-unweighted']=result['legacy_unweighted']
        for label,variant in variants.items():
            for cohort_name,cohort in cohorts.items():
                data=get('rankings/'+label+'-'+cohort_name+'.jsonl')
                if sha256(data).hexdigest()!=variant[cohort_name]['rankings_sha256']:raise ValueError('Ranking reference mismatch')
                rows=[json.loads(line) for line in data.splitlines()]
                if [r['query_id'] for r in rows]!=[q['query_id'] for q in cohort]:raise ValueError('Question alignment mismatch')
                for row,q in zip(rows,cohort,strict=True):
                    if q['language']!='tl' or row['gold_parent_id']!=q['gold_parent_id']:raise ValueError('Cohort changed')
                    ranking=row['ranking']
                    if len(ranking)!=10 or len({r[0] for r in ranking})!=10:raise ValueError('Not ten unique parents')
                    if score(q,ranking,corpus['records'])!=row['metrics']:raise ValueError('Per-question metric mismatch')
                    verified+=1
                actual=aggregate(rows)
                for key in (*METRICS,'questions','denominators'):
                    if actual[key]!=variant[cohort_name][key]:raise ValueError('Aggregate mismatch: '+label+' '+key)
    receipt={**summary,'verified_question_records':verified,'models':17,
             'verification_scope':'Full downloaded report/evidence/ranking SHA256; recomputed every per-question metric and aggregate. No independent embedding inference rerun or download of HF model weights.',
             'latency':'Not measured; previous results untouched','credits':'unverified',
             'resource_release':'Managed training terminal Completed; no endpoint or persistent EC2/EBS created by this job'}
    with (target/'verification.json').open('x') as f:json.dump(receipt,f,indent=2,default=str)
    lines=['# Complete quality repair — Tagalog held-out results','',
           '3,984 questions; 40,152 chunks; window 192 / stride 128; 1,294 parent articles. No latency remeasurement.','',
           '| Model | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Model files (MB) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in plan['order']:
        r=report['results'][name];m=r['heldout']
        lines.append('| '+name+' | '+' | '.join(f'{100*m[k]:.4f}' for k in METRICS[:6])+f" | {r['model_file_bytes']/1e6:.3f} |")
    lines+=['','All quality scores above are percentages. Full answer-span coverage at 1/3/5/10, denominators and development scores are in complete-report.json.',
            'Model-file size is disk size, NOT peak process RAM. Existing memory/latency measurements have not been rerun.',
            'Word2Vec/fastText are new frozen reproductions of the recorded recipe; their original scoring weights were not saved.',
            'TSS primary uses official token weights. Its historical unweighted variant is separately preserved.',
            'Every question ranking and score is retained. Verification replays scoring, not embedding inference.']
    with (target/'results.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'verified':str(target),'question_records':verified},indent=2))


if __name__=='__main__':main()
