"""Read-only progress and independent recorded-ranking verification."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import struct
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
    state=sm.describe_training_job(TrainingJobName=record['job_name'])
    uri=urlparse(record['s3_results']);prefix=uri.path.strip('/')
    objects=[]
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=uri.netloc,Prefix=prefix+'/'):
        objects+=page.get('Contents',[])
    events=sorted(x['Key'] for x in objects if '/events/' in x['Key'])
    latest=json.loads(s3.get_object(Bucket=uri.netloc,Key=events[-1])['Body'].read()) if events else None
    billable=state.get('BillableTimeInSeconds')
    out={'job_name':record['job_name'],'checked_at_utc':datetime.now(timezone.utc).isoformat(),
         'status':state['TrainingJobStatus'],'secondary_status':state.get('SecondaryStatus'),
         'failure_reason':state.get('FailureReason'),'latest_event':latest,
         'saved_reports':[o['Key'] for o in objects if '/reports/' in o['Key']],
         'billable_seconds':billable,'compute_estimate_usd':billable/3600*1.315 if billable is not None else None,
         'credits_verified':False,'managed_release_on_terminal':True}
    if args.verify_complete:
        from pycon_model2vec_demo.multiwikiqa_evaluation import score,summarize
        if state['TrainingJobStatus']!='Completed':raise ValueError('Not complete')
        complete=json.loads(s3.get_object(Bucket=uri.netloc,Key=prefix+'/reports/mixed-completion.json')['Body'].read())
        plan=json.loads((args.run_directory/'mixed-plan.json').read_text())
        if complete['source_plan_sha256']!=sha256((args.run_directory/'mixed-plan.json').read_bytes()).hexdigest():
            raise ValueError('Frozen plan mismatch')
        def fetch_spec(spec):
            raw=s3.get_object(Bucket=uri.netloc,Key=spec['key'])['Body'].read()
            if len(raw)!=spec['bytes'] or sha256(raw).hexdigest()!=spec['sha256']:
                raise ValueError('Independent full report hash mismatch')
            return json.loads(raw)
        def fetch(relative):return fetch_spec({**complete['artifacts'][relative],'key':prefix+'/'+relative})
        layout=fetch_spec(plan['evaluation_layout'])
        raw=s3.get_object(Bucket=uri.netloc,Key=plan['annotations']['key'])['Body'].read()
        if len(raw)!=plan['annotations']['bytes'] or sha256(raw).hexdigest()!=plan['annotations']['sha256']:
            raise ValueError('Annotation hash mismatch')
        queries=[json.loads(line) for line in raw.decode().splitlines()]
        if (len(queries)!=10000 or sum(q['partition']=='evaluation' for q in queries)!=7922
            or len(layout['parents'])!=1294 or len(layout['chunks'])!=6546):
            raise ValueError('Cohort/corpus counts changed')
        lookup={c['chunk_id']:i for i,c in enumerate(layout['chunks'])}
        new=['initialization','approved_plus_external','approved_articles_ranked',
             'external_wikipedia_ranked','approved_plus_external_ranked']
        for name in new:
            report=fetch('reports/'+name+'-quality.json')
            if report['shared_layout_sha256']!=plan['evaluation_layout']['sha256']:
                raise ValueError('Shared layout changed')
            if [r['query_id'] for r in report['per_query']]!=layout['query_ids']:
                raise ValueError('Question order/count changed')
            for q,r in zip(queries,report['per_query']):
                if any(q[k]!=r[k] for k in ('gold_parent_id','language','partition','exclusion_reasons')):
                    raise ValueError('Annotation changed')
                top=r['top10']
                if len(top)!=10 or len({x['parent_id'] for x in top})!=10:
                    raise ValueError('Ten unique parents required')
                metrics=score(q,[(x['parent_id'],lookup[x['chunk_id']],x['cosine']) for x in top],layout['chunks'])
                if metrics!=r['metrics']:raise ValueError('Recorded-ranking metric mismatch')
            summary=summarize(report['per_query'])
            for group,values in summary.items():
                if values!=report['summary'][group]:raise ValueError('Aggregate mismatch')
            for language in ('en','tl'):
                part=[r for r in report['per_query'] if r['language']==language and r['partition']=='evaluation']
                if summarize(part)['evaluation_partition']!=report['summary']['heldout_'+language]:
                    raise ValueError('Heldout language summary mismatch')
            if report['summary']!=complete['quality'][name]['summary']:
                raise ValueError('Completion table mismatch')
        for name,training in complete['ranking_reports'].items():
            manifest=fetch('reports/'+name+'-ranking-manifest.json')
            forbidden={q['query_id'] for q in queries if q['partition']=='evaluation'}
            if set(manifest['training_query_ids'])&forbidden or set(manifest['validation_query_ids'])&forbidden:
                raise ValueError('Heldout QA leakage')
            if (len(manifest['training_query_ids'])!=1734 or len(manifest['validation_query_ids'])!=338
                or set(manifest['heldout_query_ids'])!=forbidden):raise ValueError('Training split changed')
            history=training['history']
            if training['selected_by'].startswith('development combined nDCG@10 gain'):
                from pycon_model2vec_demo.multiwikiqa_ranking_v3 import select_checkpoint
                best=select_checkpoint(history)
            else:
                best=max(range(len(history)),key=lambda i:history[i]['nDCG@10'])
            if best!=training['selected_epoch']:raise ValueError('Wrong selected checkpoint')
            weight=f"models/{name}/epoch-{best:02d}/model.safetensors"
            spec=complete['artifacts'][weight]
            if spec['sha256']!=complete['quality'][name]['model_weights_sha256']:
                raise ValueError('Selected model checksum mismatch')
            head=s3.head_object(Bucket=uri.netloc,Key=prefix+'/'+weight)
            if head['ContentLength']!=spec['bytes'] or head['Metadata'].get('sha256')!=spec['sha256']:
                raise ValueError('Model S3 checksum header mismatch')
            def range_bytes(a,b):
                return s3.get_object(Bucket=uri.netloc,Key=prefix+'/'+weight,Range=f'bytes={a}-{b}')['Body'].read()
            length=struct.unpack('<Q',range_bytes(0,7))[0]
            header=json.loads(range_bytes(8,7+length))
            if not any(v.get('shape')==[249999,1024] and v.get('dtype')=='F32' for k,v in header.items() if k!='__metadata__'):
                raise ValueError('Selected weights dimension/precision mismatch')
        out['independent_verification']={'new_reports':5,'recorded_rankings_recomputed':50000,
            'report_full_bytes_hashed':True,'model_bytes_rehashed_locally':False,
            'model_headers_and_worker_checksums_checked':True,'independent_vector_reranking':False,
            'development_only_ranking':True,'heldout_questions':7922,'managed_compute_released':True}
        out['results']=complete
    print(json.dumps(out,indent=2,sort_keys=True,default=str))


if __name__=='__main__':main()
