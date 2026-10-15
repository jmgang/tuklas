"""Recover the completed mixture and repair three retrieval-ranking arms.

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
    # Actual CUDA regression gate before fetching large models. This toy fixture
    # is a runtime test only, not data used to train any experimental student.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss,MatryoshkaLoss
    from multiwikiqa_ranking import normalized_training_embeddings
    tokenizer=Tokenizer(WordLevel({'[UNK]':0,'hello':1,'world':2,'bansa':3,'Pilipinas':4},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=Whitespace()
    module=StaticEmbedding(tokenizer,embedding_weights=torch.randn(5,1024))
    testmodel=SentenceTransformer(modules=[module],device='cuda')
    output=normalized_training_embeddings(testmodel,['hello world','bansa Pilipinas'])
    criterion=MatryoshkaLoss(testmodel,MultipleNegativesRankingLoss(testmodel),PLAN['ranking']['matryoshka_dimensions'])
    features=[{k:v.to(testmodel.device) for k,v in testmodel.tokenize(texts).items()} for texts in
              (['hello','bansa'],['hello world','bansa Pilipinas'])]
    loss=criterion(features,None)
    loss.backward()
    if output.shape!=(2,1024) or not np.isfinite(output).all() or not torch.isfinite(loss):
        raise ValueError('CUDA parity/backward regression gate failed')
    preserve.event('cuda-regression-gate-passed',forward=True,backward=True)
    del testmodel,module,criterion,features,loss
    gc.collect();torch.cuda.empty_cache()
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
    report = json.loads(get(PLAN['recovered_training_report']))
    if (report['dimension']!=1024 or not report['reload_parity'] or not report['no_pca']
            or report['initial_hashes']['model.safetensors']!='a58a9d3b0cd6d9682348520797fe9da23bd450e02d1bd28a214e8eb8a6f9bcef'):
        raise ValueError('Completed mixture training provenance changed')
    original.write_json(original.WORK/'recovered-training-report.json',report)
    preserve.put(original.WORK/'recovered-training-report.json','reports/recovered-training-report.json')
    path=original.WORK/'approved_plus_external';path.mkdir(exist_ok=False)
    for filename,spec in PLAN['recovered_mixture'].items():
        response=preserve.s3.get_object(Bucket=bucket,Key=spec['key'])
        h=sha256()
        with (path/filename).open('xb') as f:
            for block in iter(lambda:response['Body'].read(2**20),b''):
                h.update(block);f.write(block)
        response['Body'].close()
        if (path/filename).stat().st_size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:
            raise ValueError('Recovered mixture full-byte checksum mismatch')
    if original.digest(path/'model.safetensors')!=PLAN['recovered_quality']['approved_plus_external']['model_weights_sha256']:
        raise ValueError('Recovered mixture differs from completed quality report')
    preserve.event('completed-mixture-restored',training_repeated=False,all_bytes_hashed=True)
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
    for name,spec in PLAN['recovered_quality'].items():
        raw=get(spec)
        previous=json.loads(raw)
        if (previous['shared_layout_sha256']!=PLAN['evaluation_layout']['sha256']
                or [r['query_id'] for r in previous['per_query']]!=layout['query_ids']):
            raise ValueError('Recovered quality provenance changed')
        out=original.WORK/f'{name}-quality.json'
        with out.open('xb') as f:f.write(raw)
        preserve.put(out,'reports/'+out.name)
        quality[name]={k:v for k,v in previous.items() if k!='per_query'}
        full_results[name]=previous['per_query']
        preserve.event('completed-quality-recovered',model=name,scoring_repeated=False)
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
         'ranking_finetuning':True,'completed_mixture_reused':True})
    preserve.put(original.WORK/'mixed-completion.json','reports/mixed-completion.json')
    preserve.event('ranking-repair-job-complete',tokenlearn_arms_retrained=0,ranking_arms=3,diagnostic_models=5)


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
