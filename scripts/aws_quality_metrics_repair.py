#!/usr/bin/env python3
"""Immutable 17-model quality repair; deliberately contains no latency benchmark."""
from datetime import datetime, timezone
from hashlib import sha256, md5
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

INPUT = Path('/opt/ml/input/data/input')
WORK = Path('/opt/ml/quality-repair')
sys.path.insert(0, str(INPUT))


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def publish(path, relative, plan):
    import boto3
    with path.open('rb') as f:
        boto3.client('s3').put_object(Bucket=plan['bucket'],
            Key=plan['output_prefix'] + relative, Body=f,
            ContentLength=path.stat().st_size, IfNoneMatch='*',
            Metadata={'sha256': digest(path)}, ServerSideEncryption='AES256')


def event(stage, plan, **details):
    value = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    print(json.dumps(value), flush=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + stage + '.json'
    path = WORK / name
    write_json(path, value)
    publish(path, 'events/' + name, plan)


def child(name):
    import boto3
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from multiwikiqa_evaluation import normalized
    from quality_metrics_complete import evaluate, legacy_unweighted_encoder
    import aws_runtime_comparison_v2 as prior
    plan = json.loads((INPUT / 'plan.json').read_text())
    spec = plan['systems'][name]
    corpus = json.loads((WORK / 'corpus.json').read_text())
    cohorts = {k: json.loads((WORK / (k + '.json')).read_text()) for k in ('development', 'heldout')}
    documents = [r['text'] for r in corpus['records']]
    folder = WORK / 'models' / name
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    runtime = spec['runtime']
    extra = {'protocol_spec': spec, 'latency_measured': False}
    if runtime == 'static-cpu':
        from model2vec import StaticModel
        if 's3_objects' in spec:
            s3 = boto3.client('s3')
            for relative, expected in spec['s3_objects'].items():
                path = folder / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(plan['bucket'], expected['key'], str(path))
                if 'sha256' in expected:
                    if digest(path) != expected['sha256']:
                        raise ValueError('Pruned package SHA256 mismatch: ' + relative)
                elif '-' not in expected['etag']:
                    if md5(path.read_bytes()).hexdigest()!=expected['etag']:
                        raise ValueError('Pruned package ETag mismatch: ' + relative)
                else:
                    raise ValueError('No usable frozen checksum for '+relative)
        else:
            snapshot_download(repo_id=spec['model']['id'], revision=spec['model']['revision'],
                local_dir=folder, allow_patterns=['config.json','model.safetensors','tokenizer.json','modules.json','README.md'])
        model = StaticModel.from_pretrained(str(folder))
        prefix = json.loads((folder / 'config.json').read_text()).get('query_prefix', '') if spec.get('query_prefix_from_config') else ''
        def enc(values, query=False):
            return model.encode([prefix+v for v in values] if query else values,
                batch_size=1024, max_length=None, use_multiprocessing=False)
        dimension = int(model.embedding.shape[1])
        extra['inference'] = 'Official Model2Vec.encode, including any shipped token weights'
    elif runtime in ('transformer-gpu', 'st-static-cpu'):
        from sentence_transformers import SentenceTransformer
        ms = spec['model']
        model = SentenceTransformer(ms['id'], revision=ms['revision'],
            device='cuda' if runtime == 'transformer-gpu' else 'cpu',
            cache_folder=str(WORK / 'hf-cache'), trust_remote_code=False)
        if runtime == 'transformer-gpu':
            model.max_seq_length = int(ms.get('max_seq_length', 512))
        def enc(values, query=False):
            prefix = ms.get('query_prefix' if query else 'document_prefix', '')
            if runtime == 'transformer-gpu':
                # Fail rather than silently truncate any benchmark input.
                for start in range(0, len(values), 512):
                    ids = model.tokenizer([prefix+v for v in values[start:start+512]],
                        truncation=False, padding=False)['input_ids']
                    if any(len(x) > model.max_seq_length for x in ids):
                        raise ValueError('Input exceeds pinned model context; no silent truncation')
            return model.encode(values, batch_size=int(ms.get('batch_size',64)),
                prompt=prefix, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
        dimension = int(model.get_sentence_embedding_dimension())
        extra['torch_dtype'] = str(next(model.parameters()).dtype)
        # Hash the actual downloaded snapshot, not only its requested revision.
        snapshots = list((WORK / 'hf-cache').glob('models--*/snapshots/' + ms['revision']))
        if len(snapshots) != 1:
            raise ValueError('Cannot identify pinned model snapshot')
        folder = snapshots[0]
    elif runtime == 'gensim-cpu':
        import gzip
        from gensim.models import FastText, Word2Vec
        recipe = spec['recipe']
        with gzip.open(INPUT / 'training.tl.jsonl.gz', 'rt') as f:
            sentences = [prior.words(json.loads(line)['text']) for line in f if line.strip()]
        if len(sentences) != recipe['expected_source_rows']:
            raise ValueError('Training corpus row count changed')
        builder = FastText if recipe['kind'] == 'fasttext' else Word2Vec
        model = builder(sentences, vector_size=recipe['dimension'], sg=recipe['sg'],
            window=recipe['window'], min_count=recipe['min_count'], workers=recipe['workers'],
            epochs=recipe['epochs'], seed=recipe['seed'])
        del sentences
        kv = model.wv
        kv.save(str(folder / 'vectors.kv'))
        dimension = int(kv.vector_size)
        def enc(values, query=False):
            out = np.zeros((len(values), dimension), dtype=np.float32)
            for i, text in enumerate(values):
                vectors = [kv[w] for w in prior.words(text) if w in kv]
                if vectors:
                    out[i] = np.mean(vectors, axis=0)
            return normalized(out)
        extra['reproduced_not_original_weights'] = True
        extra['latency_model_identity'] = 'Different reproduced weights; previous timing is recipe-level, not exact artifact'
    else:
        raise ValueError(runtime)
    manifest = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and '.cache' not in path.parts:
            rel = str(path.relative_to(folder))
            manifest[rel] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            if runtime == 'gensim-cpu':
                publish(path, 'models/' + name + '/' + rel, plan)
    extra.update({'model_files':manifest, 'model_file_bytes':sum(x['bytes'] for x in manifest.values()),
                  'dimension':dimension, 'kind':'transformer' if runtime=='transformer-gpu' else 'static'})
    event('encoding', plan, model=name)
    D = normalized(enc(documents))
    result = {**extra, 'document_vector_bytes': D.nbytes, 'document_vectors_sha256':sha256(D.tobytes()).hexdigest()}
    def run_variant(label, document_vectors, encoder):
        variant = {}
        for cohort_name, cohort in cohorts.items():
            Q = normalized(encoder([q['question'] for q in cohort], query=True))
            summary, rows = evaluate(document_vectors,Q,cohort,corpus['owners'],corpus['parents'],corpus['records'])
            path = WORK / (label + '-' + cohort_name + '.jsonl')
            with path.open('x') as f:
                for row in rows:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
            publish(path,'rankings/'+path.name,plan)
            variant[cohort_name] = {**summary, 'rankings_sha256':digest(path),
                                     'query_vectors_sha256':sha256(Q.tobytes()).hexdigest()}
        return variant
    result.update(run_variant(name,D,enc))
    expected=spec['expected_development']
    result['historical_development_comparison']={'previous':expected,
        'current':result['development']['Recall@10'],
        'delta':result['development']['Recall@10']-expected,
        'interpretation':'Audit only; never select or retune models on these scores'}
    if name=='tuklas-pruned' and abs(result['development']['Recall@10']-expected)>1e-6:
        raise ValueError('Intact Tuklas fails known development anchor; rankings preserved')
    if name=='tss-m2v-bge-m3-1024d':
        from model2vec.train.utils import get_probable_pad_token_id
        legacy=legacy_unweighted_encoder(model,get_probable_pad_token_id)
        result['legacy_unweighted'] = run_variant(name+'-legacy-unweighted',legacy(documents),legacy)
    path=WORK/(name+'.json')
    write_json(path,result)
    publish(path,'models-results/'+path.name,plan)
    event('model-complete',plan,model=name,heldout=result['heldout']['Recall@10'])


def main():
    plan=json.loads((INPUT/'plan.json').read_text())
    for filename, expected in plan['staged_sha256'].items():
        if digest(INPUT/filename)!=expected:
            raise ValueError('Input hash mismatch: '+filename)
    WORK.mkdir(parents=True,exist_ok=False)
    print('Verified staged sources; installing pinned dependencies',flush=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--quiet',
        'model2vec==0.9.0','sentence-transformers==5.7.0','transformers==5.3.0',
        'huggingface-hub==1.33.0','numpy==2.5.3','boto3==1.43.108','gensim==4.4.0'],check=True)
    event('setup-complete',plan)
    from aws_runtime_comparison_v2 import build_index
    from multiwikiqa_ranking_v5 import prepare
    layout=json.loads((INPUT/'shared-layout.json').read_text())
    records,owners=build_index(layout)
    annotations=[json.loads(line) for line in (INPUT/'annotations.all.jsonl').read_text().splitlines() if line.strip()]
    tagalog=[q for q in annotations if q['language']=='tl']
    _,dev,_=prepare(tagalog,layout['chunks'])
    held=[q for q in tagalog if q['partition']=='evaluation']
    if len(dev)!=197 or len(held)!=3984 or set(q['query_id'] for q in dev)&set(q['query_id'] for q in held):
        raise ValueError('Cohort mismatch')
    for label,value in [('corpus',{'records':records,'owners':owners,'parents':layout['parents']}),('development',dev),('heldout',held)]:
        path=WORK/(label+'.json'); write_json(path,value); publish(path,'evidence/'+path.name,plan)
    env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
    failures={}; results={}
    for name in plan['order']:
        event('model-start',plan,model=name)
        try:
            subprocess.run([sys.executable,str(INPUT/'aws_quality_metrics_repair.py'),'--child',name],env=env,check=True,timeout=2400)
            results[name]=json.loads((WORK/(name+'.json')).read_text())
        except Exception:
            failures[name]=traceback.format_exc()
            event('model-failed',plan,model=name,error=failures[name])
    report={'status':'complete' if not failures and len(results)==17 else 'incomplete',
        'results':results,'failures':failures,'protocol':plan['protocol'],
        'versions':{p:importlib.metadata.version(p) for p in ['model2vec','sentence-transformers','transformers','torch','numpy','gensim']}}
    path=WORK/'complete-report.json'; write_json(path,report); publish(path,path.name,plan)
    event('finished',plan,status=report['status'],completed=len(results))
    if report['status']!='complete': raise SystemExit('Incomplete results preserved; do not claim completion')


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--child': child(sys.argv[2])
    else: main()
