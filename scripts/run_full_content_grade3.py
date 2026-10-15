"""Frozen full-content DisastIR comparison. No deletion or artifact overwrites.

Audit all six tokenizers before any paid encoding. Local workers are isolated
processes. Old vectors are reused only for unchanged, complete single segments.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import ssl
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
if os.environ.get('COHERE_BOTO3_PATH'):
    sys.path.insert(0, os.environ['COHERE_BOTO3_PATH'])
import numpy as np
from run_cohere_grade3 import (read, save, digest, file_digest, event, prepare,
                              encode_batches, render)
from pycon_model2vec_demo.segmented_encoding import SegmentPolicy, pool
from pycon_model2vec_demo.model_registry import MODEL_NAMES, local_model_artifacts
from pycon_model2vec_demo.retrieval import exact_top_k, open_corpus_embeddings
from pycon_model2vec_demo.grade3 import TOP10_METRICS, score_top10_query, summarize, paired_with_teacher
from pycon_model2vec_demo.measurement import _timing_summary, host_metadata, peak_rss_bytes
from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere

TOKENIZER_REVISION = 'a73b09960122f77a78083ff79084162793ed9c39'
TOKENIZER_REPO = 'CohereLabs/Cohere-embed-english-v3.0'


def tokenizer_snapshot():
    import certifi
    target = ROOT / 'models' / ('cohere-v3-tokenizer-' + TOKENIZER_REVISION[:12])
    target.mkdir(parents=True, exist_ok=True)
    files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'config.json']
    for name in files:
        path = target / name
        if not path.exists():
            url = f'https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/{name}'
            data = urllib.request.urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())).read()
            with path.open('xb') as out:
                out.write(data)
    hashes = {name: file_digest(target / name) for name in files}
    save(target / 'snapshot-provenance.json', dict(repo=TOKENIZER_REPO, revision=TOKENIZER_REVISION, files_sha256=hashes))
    return target, hashes


def policies(name, artifacts, cohere_path):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    if name == 'cohere_v3':
        tokenizer = AutoTokenizer.from_pretrained(str(cohere_path), local_files_only=True)
        tokenizer.backend_tokenizer.no_truncation()
        count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))
        content = lambda text: len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        # Conservative 480 incl. specials, allowing room for provider role tokens.
        return {role: SegmentPolicy(count, content, 480, 2048) for role in ('corpus','queries','development')}
    artifact = artifacts[name]
    if name == 'teacher':
        tokenizer = AutoTokenizer.from_pretrained(str(artifact.path), local_files_only=True)
        tokenizer.backend_tokenizer.no_truncation()
        count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))
        content = lambda text: len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        chars = None
    else:
        tokenizer = Tokenizer.from_file(str(artifact.path / 'tokenizer.json'))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        unknown = tokenizer.token_to_id('[UNK]')
        count = lambda text: sum(i != unknown for i in tokenizer.encode(text, add_special_tokens=False).ids)
        content = count
        chars = 512 * int(np.median([len(token) for token in tokenizer.get_vocab()]))
    return {role: SegmentPolicy(count, content, 512, chars,
        artifact.settings['passage_prefix' if role == 'corpus' else 'query_prefix'])
        for role in ('corpus','queries','development')}


def audit(name, texts, ids, policy, directory, provenance):
    path = directory / name / 'audit.json'
    if path.exists():
        result = read(path)
        if result['provenance'] != provenance or result['ids_sha256'] != digest(ids) or result['texts_sha256'] != digest(texts):
            raise ValueError(f'Audit provenance mismatch: {path}')
        return result
    plans = []
    for i, text in enumerate(texts):
        plans.append(policy.plan(text))
        if (i+1) % 40000 == 0:
            event(directory.parent, dict(stage='preflight-audit', model=name, role=directory.name, audited=i+1,total=len(texts)))
    result = dict(provenance=provenance, ids_sha256=digest(ids), texts_sha256=digest(texts),
        plans=plans, token_limit=policy.token_limit, character_limit=policy.character_limit,
        prefix=policy.prefix, input_count=len(texts), segment_count=sum(map(len,plans)),
        split_inputs=sum(len(plan)>1 for plan in plans),
        source_text_sha256=[digest(text) for text in texts],
        semantics='Nonoverlapping original-character slices; full reconstruction verified; token-weighted mean of normalized segment vectors, then L2 normalization. Prefix and special tokens excluded from weights. Whitespace-only slices have zero weight.')
    save(path,result)
    return result


def publish_matrix(target, values, provenance):
    manifest_path = target.with_suffix('.json')
    if target.exists():
        manifest = read(manifest_path)
        if manifest['provenance'] != provenance or file_digest(target) != manifest['vectors_sha256']:
            raise ValueError(f'Published matrix differs: {target}')
        return np.load(target,mmap_mode='r')
    if manifest_path.exists():
        raise FileExistsError(f'Orphan manifest preserved: {manifest_path}')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as out:
        np.save(out,np.asarray(values,dtype=np.float32),allow_pickle=False)
        out.flush()
        os.fsync(out.fileno())
    save(manifest_path,dict(provenance=provenance,shape=list(values.shape),vectors_sha256=file_digest(target)))
    return np.load(target,mmap_mode='r')


class CompleteEncoder:
    def __init__(self, base, policy_by_role, remote=False):
        self.base, self.policies, self.remote = base, policy_by_role, remote
        self.requests=[]

    def encode(self,texts,role,plans=None):
        policy=self.policies[role]
        plans=plans if plans is not None else [policy.plan(text) for text in texts]
        pieces, owners, weights = [],[],[]
        for i,(text,plan) in enumerate(zip(texts,plans)):
            if ''.join(text[a:b] for a,b,_ in plan)!=text:
                raise ValueError('Audit does not reconstruct original text')
            for a,b,weight in plan:
                piece=text[a:b]
                if not policy.fits(piece):
                    raise ValueError('Audited slice exceeds tokenizer budget')
                if weight:
                    pieces.append(piece); owners.append(i); weights.append(weight)
        if self.remote:
            batches=[]
            for start in range(0,len(pieces),96):
                vectors,request=self.base.encode(pieces[start:start+96], 'search_document' if role=='corpus' else 'search_query')
                if request['character_truncated_inputs']:
                    raise AssertionError('Provider input was altered')
                batches.append(vectors); self.requests.append(request)
            dimension=self.base.model['dimension']
            encoded=np.concatenate(batches) if batches else np.empty((0,dimension),dtype=np.float32)
        else:
            dimension=self.base.dimension
            # Audit guarantees fit; static native truncation is disabled too.
            if hasattr(self.base,'query_prefix'):
                encoded=np.asarray(self.base.model.encode([policy.prefix+p for p in pieces],max_length=None,batch_size=self.base.batch_size,use_multiprocessing=False),dtype=np.float32)
            else:
                encoded=(self.base.encode_passages if role=='corpus' else self.base.encode_queries)(pieces)
        result=np.zeros((len(texts),dimension),dtype=np.float32)
        cursor=0
        for i in range(len(texts)):
            start=cursor
            while cursor<len(owners) and owners[cursor]==i:
                cursor+=1
            if cursor>start:
                result[i]=pool(encoded[start:cursor],weights[start:cursor])
        return result


def operational(wrapper,corpus,vectors,prepared,directory,name,provenance,artifact=None):
    path=directory/name/'measurement.json'
    if path.exists():
        record=read(path)
        if record['provenance']!=provenance: raise ValueError('Measurement protocol differs')
        return record
    old=read(next(ROOT.glob('runs/measurement-teacher-*.json')))
    ids=old['development_query_ids']
    lookup={q.query_id:q.query for q in prepared.development}
    samples,searches,total=[],[],[]
    for round_index in range(7):
        for query_id in ids:
            before=time.perf_counter()
            query=wrapper.encode([lookup[query_id]],'development')
            encoded=time.perf_counter()
            exact_top_k(query,vectors,corpus.ids,k=10,corpus_block_size=4096)
            after=time.perf_counter()
            if round_index>=2:
                samples.append(encoded-before); searches.append(after-encoded); total.append(after-before)
        event(directory,dict(stage='operational-measurement',model=name,round=round_index+1,rounds=7))
    record=dict(provenance=provenance,environment=host_metadata(),development_query_ids=ids,
        warm=dict(query_count=len(ids),warmup_rounds=2,repeated_rounds=5,
                  single_query_encoding=_timing_summary(samples),single_query_exact_search=_timing_summary(searches),single_query_end_to_end=_timing_summary(total)),
        model_artifact_size_bytes=sum((artifact.path/key).stat().st_size for key in artifact.files_sha256) if artifact else None,
        peak_rss_after_workload_bytes=peak_rss_bytes() if artifact else None,
        peak_rss_client_bytes=peak_rss_bytes(),
        note='Isolated local model worker RSS includes data, tokenizer, mapped vectors and runtime; not model-only RAM. Remote RAM and weights unavailable. Latency includes segmentation and pooling, and AWS network for Cohere. Same 32 development questions, 2 warmup and 5 measured rounds.')
    save(path,record)
    return record


def score(name,wrapper,corpus,queries,prepared,vectors,directory,provenance,artifact=None):
    target=directory/name/'query-vectors.npy'
    if target.exists():
        qvectors=publish_matrix(target,np.empty((0,0)),provenance)
    else:
        plans=read(directory/'audits'/'queries'/name/'audit.json')['plans']
        qvectors=publish_matrix(target,wrapper.encode([q.query for q in queries],'queries',plans),provenance)
    measurement=operational(wrapper,corpus,vectors,prepared,directory,name,provenance,artifact)
    rows=[]
    for start in range(0,len(queries),16):
        batch=queries[start:start+16]
        rankings=exact_top_k(qvectors[start:start+len(batch)],vectors,corpus.ids,k=10,corpus_block_size=4096)
        for query,ranking in zip(batch,rankings):
            rows.append(dict(query_id=query.query_id,category=query.category,raw_grade3_count=query.raw_grade3_count,
                mapped_grade3_count=len(query.gold_ids),metrics=score_top10_query(query,ranking),
                top10=[dict(corpus_id=r.corpus_id,similarity=r.similarity) for r in ranking]))
        if (start//16+1)%10==0 or len(rows)==len(queries):
            event(directory,dict(stage='quality-scoring',model=name,completed_queries=len(rows),total_queries=len(queries)))
    result=dict(provenance=provenance,per_query=rows,measurement=measurement,
        corpus_vector_size_bytes=(directory/name/'corpus-vectors.npy').stat().st_size)
    if wrapper.remote:
        result['measurement_requests']=wrapper.requests
    save(directory/name/'query-results.json',result)
    event(directory,dict(stage='model-complete',model=name,metrics=summarize(rows,metrics=TOP10_METRICS)['overall']))


def local_worker(name,config,directory,provenance):
    corpus,queries,prepared,*_=prepare(config)
    artifacts=local_model_artifacts(ROOT)
    artifact=artifacts[name]
    base=artifact.load_encoder()
    wrapper=CompleteEncoder(base,policies(name,artifacts,None))
    target=directory/name/'corpus-vectors.npy'
    if target.exists():
        vectors=publish_matrix(target,np.empty((0,0)),provenance)
    else:
        original=ROOT/'runs'/'embeddings'/f'{name}-{artifact.fingerprint[:12]}.npy'
        cached=open_corpus_embeddings(original,corpus=corpus,corpus_sha256=config['corpus_sha256'],model_fingerprint=artifact.fingerprint)
        plans=read(directory/'audits'/'corpus'/name/'audit.json')['plans']
        # Complete single segments have exactly the same original input.
        changed=[i for i,plan in enumerate(plans) if len(plan)!=1]
        values=np.array(cached,dtype=np.float32,copy=True)
        for start in range(0,len(changed),128):
            selected=changed[start:start+128]
            values[selected]=wrapper.encode([corpus.texts[i] for i in selected],'corpus',[plans[i] for i in selected])
            event(directory,dict(stage='local-full-content-repair',model=name,completed=min(start+128,len(changed)),total_changed=len(changed),reused_complete_rows=len(plans)-len(changed)))
        vectors=publish_matrix(target,values,provenance)
        save(directory/name/'reuse.json',dict(source=str(original.relative_to(ROOT)),source_sha256=file_digest(original),source_model_fingerprint=artifact.fingerprint,
            reused_complete_rows=len(plans)-len(changed),recomputed_rows=changed,eligibility='Audited full original input fits both original token and character limits with role prefix.'))
        del values,cached
        gc.collect()
    score(name,wrapper,corpus,queries,prepared,vectors,directory,provenance,artifact)


def remote_worker(config,directory,provenance,corpus,queries,prepared,artifacts,tokenizer_path):
    import boto3
    from botocore.config import Config
    client=boto3.Session(profile_name=config['aws_profile']).client('bedrock-runtime',region_name=config['aws_region'],
        config=Config(connect_timeout=10,read_timeout=90,max_pool_connections=6,retries={'max_attempts':0}))
    settings={**config['models']['cohere_v3'],'truncate':'NONE','maximum_input_characters':2048,'full_content':True}
    encoder=BedrockCohere(client,settings,tokens_per_minute=config['tokens_per_minute'])
    wrapper=CompleteEncoder(encoder,policies('cohere_v3',artifacts,tokenizer_path),remote=True)
    target=directory/'cohere_v3'/'corpus-vectors.npy'
    if target.exists():
        values=publish_matrix(target,np.empty((0,0)),provenance)
    else:
        plans=read(directory/'audits'/'corpus'/'cohere_v3'/'audit.json')['plans']
        legacy=ROOT/'runs'/'cohere-grade3-3ac887ab565e'/'cohere_v3'/'search_document'
        reused={}
        # Validate every old batch checksum and original request fingerprint;
        # then reuse only rows that pass the pinned 480-token/2048-char policy.
        for meta_path in sorted(legacy.glob('*.json')):
            meta=read(meta_path); source=meta_path.with_suffix('.npy'); p=meta['provenance']
            a,b=p['start'],p['stop']
            if p['corpus_sha256']!=config['corpus_sha256'] or p['role']!='search_document' or p['model']!='cohere_v3' or p['settings']['model_id']!=settings['model_id'] or p['settings']['dimension']!=1024 or p['texts_sha256']!=digest(corpus.texts[a:b]):
                raise ValueError('Legacy Cohere provenance differs')
            if not source.is_file() or file_digest(source)!=meta['vectors_sha256']:
                raise ValueError('Legacy Cohere checksum differs')
            old=np.load(source,mmap_mode='r')
            if old.shape!=(b-a,1024) or not np.isfinite(old).all() or not np.allclose(np.linalg.norm(old,axis=1),1,atol=2e-5):
                raise ValueError('Legacy Cohere vectors invalid')
            for i in range(a,b):
                if len(plans[i])==1:
                    reused[i]=(source,i-a,meta['vectors_sha256'])
        segments,owners,weights=[],[],[]
        for i,(text,plan) in enumerate(zip(corpus.texts,plans)):
            if i in reused: continue
            for a,b,w in plan:
                if w:
                    piece=text[a:b]
                    if not wrapper.policies['corpus'].fits(piece): raise ValueError('Audited remote slice exceeds budget')
                    segments.append(piece); owners.append(i); weights.append(w)
        save(directory/'cohere_v3'/'segment-map.json',dict(provenance=provenance,owners=owners,weights=weights,
            segments_sha256=digest(segments),reused_rows={str(i):dict(source=str(p.relative_to(ROOT)),row=j,sha256=h) for i,(p,j,h) in reused.items()}))
        batch_config={k:v for k,v in config.items() if k!='reuse_verified_batches_from'}
        metas,paths=encode_batches('cohere_v3','search_document',tuple(segments),encoder,batch_config,directory,
            {**provenance,'settings':settings,'model':'cohere_v3','query_ids_sha256':digest([q.query_id for q in queries]),'aws_profile':config['aws_profile'],'aws_region':config['aws_region']})
        all_segments=np.concatenate([np.load(path,mmap_mode='r') for path in paths])
        values=np.zeros((len(corpus.ids),1024),dtype=np.float32)
        for source in sorted({item[0] for item in reused.values()}):
            old=np.load(source,mmap_mode='r')
            for i,(p,j,_) in reused.items():
                if p==source: values[i]=old[j]
        cursor=0
        for i in range(len(corpus.ids)):
            start=cursor
            while cursor<len(owners) and owners[cursor]==i: cursor+=1
            if cursor>start: values[i]=pool(all_segments[start:cursor],weights[start:cursor])
        values=publish_matrix(target,values,provenance)
        requests=[meta['request'] for meta in metas]
        save(directory/'cohere_v3'/'encoding-cost.json',dict(new_input_tokens=sum(r['input_tokens'] or 0 for r in requests),
            new_estimated_cost_usd=sum(r['input_tokens'] or 0 for r in requests)/1e6*.1,
            new_segments=len(segments),reused_complete_passages=len(reused),
            note='Additional invocation estimate only; preserved historical spend and credits not confirmed.'))
        del all_segments
        gc.collect()
    score('cohere_v3',wrapper,corpus,queries,prepared,values,directory,provenance)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local',choices=MODEL_NAMES)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/'configs/cohere-grade3-v2.json')
    protocol=dict(base_config=config,policy='segmented-full-content-v1',pooling='content-token-weighted mean of normalized segments; L2 normalization',
        cohere_v3_token_limit=480,cohere_v3_character_limit=2048,provider_truncate='NONE',
        cohere_tokenizer=dict(repo=TOKENIZER_REPO,revision=TOKENIZER_REVISION),
        source_files_sha256={path:file_digest(ROOT/path) for path in ['scripts/run_full_content_grade3.py','src/pycon_model2vec_demo/segmented_encoding.py','src/pycon_model2vec_demo/bedrock_embeddings.py','scripts/run_cohere_grade3.py']})
    protocol_hash=digest(protocol)
    directory=ROOT/'runs'/('full-content-grade3-'+protocol_hash[:12])
    directory.mkdir(parents=True,exist_ok=True)
    provenance=dict(protocol_sha256=protocol_hash,corpus_sha256=config['corpus_sha256'])
    save(directory/'protocol.json',protocol)
    if args.local:
        local_worker(args.local,config,directory,provenance)
        return
    lock=(directory/'runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    event(directory,dict(stage='preparing-full-content',pid=os.getpid()))
    tokenizer_path,hashes=tokenizer_snapshot()
    artifacts=local_model_artifacts(ROOT)
    corpus,queries,prepared,_,selection,reference=prepare(config)
    measurement_ids=read(next(ROOT.glob('runs/measurement-teacher-*.json')))['development_query_ids']
    dev_lookup={q.query_id:q.query for q in prepared.development}
    roles={'corpus':(corpus.texts,corpus.ids),'queries':(tuple(q.query for q in queries),tuple(q.query_id for q in queries)),
           'development':(tuple(dev_lookup[i] for i in measurement_ids),measurement_ids)}
    audits={}
    for name in (*MODEL_NAMES,'cohere_v3'):
        specs=policies(name,artifacts,tokenizer_path)
        model_provenance={**provenance,'model':name,'artifact_fingerprint':artifacts[name].fingerprint if name in artifacts else None,
            'tokenizer_hashes':hashes if name=='cohere_v3' else {key:value for key,value in artifacts[name].files_sha256.items() if 'tokenizer' in key or 'vocab' in key}}
        audits[name]={}
        for role,(texts,ids) in roles.items():
            result=audit(name,texts,ids,specs[role],directory/'audits'/role,model_provenance)
            audits[name][role]={k:result[k] for k in ('input_count','segment_count','split_inputs','token_limit','character_limit','prefix')}
        event(directory,dict(stage='model-audit-complete',model=name,audit=audits[name]))
    save(directory/'audit-summary.json',dict(provenance=provenance,models=audits,query_count=len(queries),passage_count=len(corpus.ids)))
    event(directory,dict(stage='all-preflights-complete',audit=audits))
    if args.audit_only: return
    for name in MODEL_NAMES:
        if not (directory/name/'query-results.json').exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--local',name],check=True,cwd=ROOT,env=os.environ.copy())
    if not (directory/'cohere_v3'/'query-results.json').exists():
        remote_worker(config,directory,provenance,corpus,queries,prepared,artifacts,tokenizer_path)
    models,rows={},{}
    for name in (*MODEL_NAMES,'cohere_v3'):
        result=read(directory/name/'query-results.json')
        if result['provenance']!=provenance: raise ValueError('Result protocol mismatch')
        rows[name]=result['per_query']
        models[name]=dict(status='completed',quality=summarize(rows[name],metrics=TOP10_METRICS),measurement=result['measurement'],
            corpus_vector_size_bytes=result['corpus_vector_size_bytes'])
        if name=='cohere_v3':
            cost=read(directory/name/'encoding-cost.json')
            models[name]['operational']=dict(input_tokens=cost['new_input_tokens'],estimated_api_cost_usd=cost['new_estimated_cost_usd'])
    models['cohere_v4']=dict(status='blocked',error='Previously verified AWS Marketplace INVALID_PAYMENT_INSTRUMENT; agreement NOT_AVAILABLE. No new v4 invocation attempted.')
    report=dict(status='completed-with-blocked-models',query_count=len(queries),corpus_passage_count=len(corpus.ids),top_k=10,
        corpus_sha256=config['corpus_sha256'],protocol_sha256=protocol_hash,policy=protocol['policy'],audits=audits,
        selection_audit=selection,reference_cohort_sha256=reference['provenance']['cohort_sha256'],
        models=models,paired_with_teacher={name:paired_with_teacher(rows['teacher'],r,metrics=TOP10_METRICS) for name,r in rows.items() if name!='teacher'})
    path=directory/'reports'/'full-content-report.json'
    save(path,report)
    md=path.with_suffix('.md')
    text=render(report)+'\nFull-content policy: nonoverlapping lossless slices, model-specific token/character budgets, token-weighted mean and L2 normalization. This changes retrieval behavior; it is not native long-context encoding. Static segment encoding disables the adapter cap after fit validation. All six accessible models were rescored; historical quality metrics are not reused. RSS is whole isolated-worker memory, not weights-only memory. Cohere cost covers new corpus segments, not all historical and query calls.\n'
    if md.exists():
        if md.read_text()!=text: raise ValueError('Existing Markdown report differs')
    else:
        with md.open('x') as out: out.write(text)
    event(directory,dict(stage='report-complete',report=str(path.relative_to(ROOT))))


if __name__=='__main__':
    main()
