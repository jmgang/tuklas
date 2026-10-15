"""Read-only progress/independent report verification; no large model downloads."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.parse import urlparse
import boto3

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory',type=Path)
    parser.add_argument('--verify-complete',action='store_true')
    args=parser.parse_args()
    record=json.loads((args.run_directory/'resource-record.json').read_text())
    session=boto3.Session(profile_name=record['profile'],region_name=record['region'])
    sm,s3=session.client('sagemaker'),session.client('s3')
    job=sm.describe_training_job(TrainingJobName=record['job_name'])
    uri=urlparse(record['s3_results'])
    prefix=uri.path.strip('/')
    objects=[]
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=uri.netloc,Prefix=prefix+'/'):
        objects.extend(page.get('Contents',[]))
    events=sorted(o['Key'] for o in objects if '/events/' in o['Key'])
    latest=json.loads(s3.get_object(Bucket=uri.netloc,Key=events[-1])['Body'].read()) if events else None
    billable=job.get('BillableTimeInSeconds')
    out={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'job_name':record['job_name'],
        'status':job['TrainingJobStatus'],'secondary_status':job.get('SecondaryStatus'),
        'failure_reason':job.get('FailureReason'),'latest_event':latest,
        'completed_model_reports':[o['Key'] for o in objects if '/reports/' in o['Key']],
        'billable_seconds':billable,'compute_estimate_usd':billable/3600*1.315 if billable else None,
        'credits_verified':False}
    if args.verify_complete:
        from pycon_model2vec_demo.multiwikiqa_evaluation import score,summarize
        if job['TrainingJobStatus']!='Completed':
            raise ValueError('Evaluation not complete')
        def fetch(name):
            raw=s3.get_object(Bucket=uri.netloc,Key=prefix+'/'+name)['Body'].read()
            if name!='reports/completion.json':
                spec=complete['artifacts'][name]
                if len(raw)!=spec['bytes'] or sha256(raw).hexdigest()!=spec['sha256']:
                    raise ValueError('Full report byte checksum mismatch: '+name)
            return json.loads(raw)
        complete=fetch('reports/completion.json')
        layout=fetch('shared-layout.json')
        combined=fetch('reports/combined.json')
        names={'teacher','approved_articles','external_wikipedia','potion_retrieval'}
        if set(complete['models'])!=names or complete['questions']!=10000:
            raise ValueError('Missing model/question cohort')
        queries=[json.loads(x) for x in (ROOT/'runs/multiwikiqa-preparation-20261004-v1/annotations.all.jsonl').read_text().splitlines()]
        if layout['query_ids']!=[q['query_id'] for q in queries] or len(layout['parents'])!=1294:
            raise ValueError('Common question/parent identity changed')
        chunks=layout['chunks']
        lookup={c['chunk_id']:i for i,c in enumerate(chunks)}
        for name in sorted(names):
            report=fetch('reports/'+name+'.json')
            if report['question_count']!=10000 or report['shared_layout_sha256']!=complete['artifacts']['shared-layout.json']['sha256']:
                raise ValueError('Report provenance mismatch')
            if [r['query_id'] for r in report['per_query']]!=layout['query_ids']:
                raise ValueError('Model reordered/dropped questions')
            for query,result in zip(queries,report['per_query']):
                top=result['top10']
                if len(top)!=10 or len({r['parent_id'] for r in top})!=10:
                    raise ValueError('Ten unique source parents required')
                for key in ('gold_parent_id','language','partition','exclusion_reasons'):
                    if result[key]!=query[key]:
                        raise ValueError('Question annotation changed')
                expected=score(query,[(r['parent_id'],lookup[r['chunk_id']],r['cosine']) for r in top],chunks)
                if result['metrics']!=expected:
                    raise ValueError('Independent metric recomputation failed')
            if summarize(report['per_query'])!=report['summary']:
                raise ValueError('Aggregate metrics failed')
            if combined['models'][name]['summary']!=report['summary']:
                raise ValueError('Combined table mismatch')
        out['independent_verification']={'models':4,'questions_per_model':10000,
            'all_metrics_recomputed':True,'full_report_bytes_hashed':True,
            'top10_unique_parents':True,'no_full_model_download':True,
            'managed_compute_released':True}
        out['results']=combined
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
