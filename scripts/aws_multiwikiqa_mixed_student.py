"""Train corpus UNION, then the three authorized retrieval-ranking arms.

No teacher encoding, new dataset or historical-output replacement.
Development-only QA ranking training; evaluation questions never train.
Also evaluate the shared initialization to audit actual training regression.
"""
import os
from pathlib import Path
import sys
import json
from hashlib import sha256
import importlib
import gc
import time

INPUT = Path('/opt/ml/input/data/input')
sys.path.insert(0, str(INPUT))
os.environ['SEALION_INPUT_DIR'] = str(INPUT)
os.environ['SEALION_WORK_DIR'] = '/opt/ml/mixed-student-work'
original = importlib.import_module('aws_sealion_train_repair_v3')
CONFIG = json.loads((INPUT/'settings.json').read_text())
PLAN = json.loads((INPUT/'mixed-plan.json').read_text())


def main():
    original.WORK.mkdir(parents=True, exist_ok=False)
    (original.WORK/'events').mkdir()
    preserve = original.Preservation()
    for name, spec in PLAN['source_files'].items():
        path = INPUT/name
        if path.stat().st_size!=spec['bytes'] or original.digest(path)!=spec['sha256']:
            raise ValueError('Staged source changed: '+name)
    preserve.event('mixed-dependency-setup')
    original.bootstrap()
    from sealion_runtime import verify_runtime
    verify_runtime(CONFIG['tokenlearn']['commit'])
    import numpy as np
    import torch
    from model2vec import StaticModel
    from multiwikiqa_evaluation import rank_parents, score, summarize, paired_cluster_ci
    from multiwikiqa_ranking import train_ranked
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for authorized training')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    bucket = PLAN['bucket']
    def get(spec):
        body = preserve.s3.get_object(Bucket=bucket, Key=spec['key'])['Body']
        try:
            raw = body.read()
        finally:
            body.close()
        if len(raw)!=spec['bytes'] or sha256(raw).hexdigest()!=spec['sha256']:
            raise ValueError('Frozen input byte hash changed: '+spec['key'])
        return raw
    initial = original.restore_initialization(preserve)
    merged_rows = []
    feature_dir = original.WORK/'approved_plus_external-features'
    feature_dir.mkdir(exist_ok=False)
    offset = 0
    article_partitions = {}
    for arm in ('external_wikipedia','approved_articles'):
        specs = PLAN['arms'][arm]
        layout = [json.loads(x) for x in get(specs['layout']).decode().splitlines()]
        if len(layout)!=specs['windows']:
            raise ValueError('Frozen source window count changed')
        for row in layout:
            if row['article_id'] in article_partitions and article_partitions[row['article_id']] != row['partition']:
                raise ValueError('Mixed article appears in both training and validation')
            article_partitions[row['article_id']] = row['partition']
        local_rows = 0
        for pair in specs['pairs']:
            texts = json.loads(get(pair['text']))
            import io
            vectors = np.load(io.BytesIO(get(pair['vectors'])), allow_pickle=False)
            if (texts != [r['text'] for r in layout[local_rows:local_rows+len(texts)]]
                    or vectors.shape!=(len(texts),1024) or vectors.dtype!=np.float32
                    or not np.isfinite(vectors).all()):
                raise ValueError('Frozen target/text alignment changed')
            text_path = feature_dir/f'features-{offset:08d}.json'
            vector_path = feature_dir/f'features-{offset:08d}.npy'
            with text_path.open('xb') as f:
                f.write(json.dumps(texts, ensure_ascii=False).encode()+b'\n')
            with vector_path.open('xb') as f:
                np.save(f, vectors, allow_pickle=False)
            merged_rows.extend(layout[local_rows:local_rows+len(texts)])
            offset += len(texts)
            local_rows += len(texts)
        if local_rows!=len(layout):
            raise ValueError('Incomplete frozen targets')
        preserve.event('mixed-arm-imported', arm=arm, rows=local_rows, cumulative_rows=offset)
    if len(merged_rows)!=182724:
        raise ValueError('Expected full unweighted corpus union')
    train = sum(r['partition']=='train' for r in merged_rows)
    if train!=164691 or len(merged_rows)-train!=18033:
        raise ValueError('Mixed partition counts changed')
    original.write_json(original.WORK/'mixture-provenance.json',
        {'plan':PLAN, 'source_rows':181919, 'windows':182724,
         'train':train, 'validation':18033, 'sampling':'simple union; no oversampling',
         'no_qa_training_inputs':True, 'same_initialization':True})
    preserve.put(original.WORK/'mixture-provenance.json','reports/mixture-provenance.json')
    original.train_arm('approved_plus_external', merged_rows, feature_dir, initial, preserve)
    report = json.loads((original.WORK/'approved_plus_external-training-report.json').read_text())
    del merged_rows
    gc.collect()
    torch.cuda.empty_cache()
    layout = json.loads(get(PLAN['evaluation_layout']))
    queries = [json.loads(x) for x in get(PLAN['annotations']).decode().splitlines()]
    if len(queries)!=10000 or len(layout['chunks'])!=6546 or len(layout['parents'])!=1294:
        raise ValueError('Approved evaluation cohort changed')
    if [q['query_id'] for q in queries]!=layout['query_ids']:
        raise ValueError('Evaluation ordering changed')
    chunks = layout['chunks']
    ids = layout['parents']
    positions = {name:i for i,name in enumerate(ids)}
    owners = [positions[c['parent_id']] for c in chunks]
    def evaluate(model, cohort):
        documents = model.encode([c['text'] for c in chunks],batch_size=512,max_length=None)
        questions = model.encode([q['question'] for q in cohort],batch_size=512,max_length=None)
        results = []
        for start in range(0,len(cohort),64):
            ranks = rank_parents(questions[start:start+64],documents,owners,ids)
            for q, ranking in zip(cohort[start:start+64],ranks):
                results.append({k:q[k] for k in ('query_id','gold_parent_id','language','partition','exclusion_reasons')}
                    | {'metrics':score(q,ranking,chunks), 'top10':[{'parent_id':parent,
                        'chunk_id':chunks[index]['chunk_id'],'cosine':cosine} for parent,index,cosine in ranking]})
        summary = summarize(results)
        for language in ('en','tl'):
            part=[r for r in results if r['language']==language and r['partition']=='evaluation']
            if part: summary['heldout_'+language]=summarize(part)['evaluation_partition']
        return {'question_count':len(cohort),'per_query':results,'summary':summary,
                'vector_bytes':documents.nbytes,'query_vector_bytes':questions.nbytes}
    quality = {}; full_results = {}
    for name,spec in PLAN['frozen_baseline_reports'].items():
        previous=json.loads(get(spec))
        if (previous['shared_layout_sha256']!=PLAN['evaluation_layout']['sha256']
                or [r['query_id'] for r in previous['per_query']]!=layout['query_ids']):
            raise ValueError('Frozen baseline evaluation provenance changed')
        quality[name]={k:v for k,v in previous.items() if k!='per_query'}
        full_results[name]=previous['per_query']
        for language in ('en','tl'):
            part=[r for r in previous['per_query'] if r['language']==language and r['partition']=='evaluation']
            quality[name]['summary']['heldout_'+language]=summarize(part)['evaluation_partition']
    def publish_quality(name,path,ranked=False):
        model = StaticModel.from_pretrained(path)
        if model.embedding.shape!=(249999,1024) or model.embedding.dtype!=np.float32:
            raise ValueError('Frozen capacity/precision changed')
        preserve.event('quality-encoding',model=name)
        artifact = {'model':name,**evaluate(model,queries),
            'shared_layout_sha256':PLAN['evaluation_layout']['sha256'],
            'model_weights_sha256':original.digest(path/'model.safetensors'),
            'model_weights_bytes':int(model.embedding.nbytes),
            'latency_measured':False,'runtime_memory_measured':False,
            'no_retrieval_pair_finetuning':not ranked}
        out = original.WORK/f'{name}-quality.json'
        original.write_json(out,artifact)
        preserve.put(out,'reports/'+out.name)
        quality[name] = {k:v for k,v in artifact.items() if k!='per_query'}
        full_results[name]=artifact['per_query']
        preserve.event('quality-complete',model=name,questions=10000)
        del model,artifact
        gc.collect()
    publish_quality('initialization',initial)
    publish_quality('approved_plus_external',original.WORK/'approved_plus_external')
    ranking_reports={}
    for arm in ('approved_articles','external_wikipedia','approved_plus_external'):
        if arm=='approved_plus_external':
            path=original.WORK/arm
        else:
            path=original.WORK/('frozen-'+arm);path.mkdir(exist_ok=False)
            for filename,spec in PLAN['frozen_students'][arm].items():
                # Stream large immutable weights; verify the complete file bytes.
                response=preserve.s3.get_object(Bucket=bucket,Key=spec['key'])
                h=sha256()
                with (path/filename).open('xb') as f:
                    for block in iter(lambda:response['Body'].read(2**20),b''):
                        h.update(block);f.write(block)
                response['Body'].close()
                if (path/filename).stat().st_size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:
                    raise ValueError('Frozen student changed')
        name=arm+'_ranked'
        preserve.event('ranking-started',model=name)
        selected,ranking_reports[name]=train_ranked(name,path,queries,chunks,evaluate,
                                                  preserve,original.WORK,PLAN['ranking'])
        publish_quality(name,selected,ranked=True)
    comparisons={}
    for arm in ('approved_articles','external_wikipedia','approved_plus_external'):
        ranked=arm+'_ranked'
        for before in (arm,'potion_retrieval'):
            for language in (None,'en','tl'):
                def eligible(r):return r['partition']=='evaluation' and (language is None or r['language']==language)
                left=[r for r in full_results[ranked] if eligible(r)]
                right=[r for r in full_results[before] if eligible(r)]
                comparisons[ranked+' minus '+before+' '+str(language)]={m:paired_cluster_ci(left,right,m)
                    for m in ('Recall@10','MRR@10','nDCG@10')}
    original.write_json(original.WORK/'mixed-completion.json',
        {'status':'complete','training':report,'quality':quality,'ranking_reports':ranking_reports,
         'heldout_comparisons':comparisons,'artifacts':preserve.index,
         'source_plan_sha256':original.digest(INPUT/'mixed-plan.json'),
         'qa_training':'development only; heldout 7922 questions excluded',
         'ranking_finetuning':True})
    preserve.put(original.WORK/'mixed-completion.json','reports/mixed-completion.json')
    preserve.event('mixed-job-complete',tokenlearn_arms=1,ranking_arms=3,diagnostic_models=5)


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
