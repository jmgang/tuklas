"""Shared segments, max-segment parent ranking, immutable DisastIR evaluation."""
from __future__ import annotations
import argparse
import fcntl
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
if os.environ.get('COHERE_BOTO3_PATH'):
    sys.path.insert(0,os.environ['COHERE_BOTO3_PATH'])
import numpy as np
from run_cohere_grade3 import read,save,digest,file_digest,event,prepare,encode_batches,render
from run_full_content_grade3 import tokenizer_snapshot,policies,publish_matrix,TOKENIZER_REVISION,TOKENIZER_REPO
from pycon_model2vec_demo.parent_retrieval import shared_slices,parent_top_k
from pycon_model2vec_demo.model_registry import MODEL_NAMES,local_model_artifacts
from pycon_model2vec_demo.retrieval import open_corpus_embeddings
from pycon_model2vec_demo.grade3 import TOP10_METRICS,score_top10_query,summarize,paired_with_teacher
from pycon_model2vec_demo.measurement import _timing_summary,host_metadata,peak_rss_bytes
from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere

NAMES=(*MODEL_NAMES,'cohere_v3')


def batch_counters(artifacts,tokenizer_path,specs):
    """Native batched tokenization; verify parity with the reference counters."""
    from tokenizers import Tokenizer
    counters={}
    for name in NAMES:
        path=tokenizer_path if name=='cohere_v3' else artifacts[name].path
        tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
        tokenizer.no_truncation(); tokenizer.no_padding()
        special=name in ('teacher','cohere_v3')
        unknown=tokenizer.token_to_id('[UNK]')
        def count(texts,t=tokenizer,special=special,unknown=unknown):
            rows=t.encode_batch(texts,add_special_tokens=special)
            return [len(row.ids) if special else sum(i!=unknown for i in row.ids) for row in rows]
        samples=['passage: Example text.','query: 日本語 café 123','A long word abcdefghijklmnop']
        if count(samples)!=[specs[name]['corpus'].count(t) for t in samples]:
            raise ValueError(f'Batched tokenizer differs from pinned reference: {name}')
        counters[name]=count
    return counters


def verified_json(path,provenance):
    value=read(path)
    check=read(path.with_name(path.name+'.sha256.json'))
    if check['sha256']!=file_digest(path) or value['provenance']!=provenance:
        raise ValueError(f'Immutable audit/result differs: {path}')
    return value


def checked_save(path,value):
    save(path,value)
    save(path.with_name(path.name+'.sha256.json'),dict(sha256=file_digest(path)))


def make_catalog(corpus,queries,prepared,artifacts,tokenizer_path,tokenizer_hashes,directory,provenance):
    path=directory/'shared-catalog.json'
    if path.exists():
        catalog=verified_json(path,provenance)
        if catalog['corpus_ids_sha256']!=digest(corpus.ids) or catalog['query_ids']!=[q.query_id for q in queries]:
            raise ValueError('Shared corpus/query order differs')
        return catalog
    specs={name:policies(name,artifacts,tokenizer_path) for name in NAMES}
    dev_ids=read(next(ROOT.glob('runs/measurement-teacher-*.json')))['development_query_ids']
    development={q.query_id:q.query for q in prepared.development}
    # Query semantics stay unchanged: every query must fit every model.
    for name in NAMES:
        for q in queries:
            if not specs[name]['queries'].fits(q.query):
                raise ValueError(f'Query {q.query_id} exceeds {name} budget; requires a separately approved query policy')
        for query_id in dev_ids:
            if not specs[name]['development'].fits(development[query_id]):
                raise ValueError(f'Development query {query_id} exceeds {name} budget')
    passage_policies=[specs[name]['corpus'] for name in NAMES]
    counters=batch_counters(artifacts,tokenizer_path,specs)
    segments=[]
    boundaries=[]
    split=0
    token_totals={name:0 for name in NAMES}
    for block_start in range(0,len(corpus.texts),512):
        block=corpus.texts[block_start:block_start+512]
        lengths={name:counters[name]([specs[name]['corpus'].prefix+t for t in block]) for name in NAMES}
        for offset,text in enumerate(block):
            owner=block_start+offset
            fits=all(lengths[name][offset]<=specs[name]['corpus'].token_limit and
                (specs[name]['corpus'].character_limit is None or len(specs[name]['corpus'].prefix+text)<=specs[name]['corpus'].character_limit) for name in NAMES)
            slices=[[0,len(text)]] if fits else shared_slices(text,passage_policies)
            boundaries.append(slices)
            split+=len(slices)>1
            for a,b in slices:
                piece=text[a:b]
                # Whitespace remains in source offsets but has no retrieval tokens.
                if piece.strip():
                    segments.append([owner,a,b])
                    for name in NAMES:
                        p=specs[name]['corpus']
                        if fits:
                            token_totals[name]+=lengths[name][offset]
                        else:
                            if not p.fits(piece): raise AssertionError('Shared slice does not fit')
                            token_totals[name]+=p.count(p.prefix+piece)
            if not any(text[a:b].strip() for a,b in slices):
                raise ValueError(f'Entire passage {corpus.ids[owner]} is whitespace')
        completed=block_start+len(block)
        if block_start//512%40==0 or completed==len(corpus.texts):
            event(directory,dict(stage='shared-layout-preflight',audited_passages=completed,total_passages=len(corpus.ids),segments=len(segments),split_passages=split))
    catalog=dict(provenance=provenance,corpus_ids_sha256=digest(corpus.ids),corpus_texts_sha256=digest(corpus.texts),
        source_text_sha256=[digest(t) for t in corpus.texts],query_ids=[q.query_id for q in queries],query_texts_sha256=digest([q.query for q in queries]),
        development_query_ids=dev_ids,development_texts_sha256=digest([development[i] for i in dev_ids]),
        passage_count=len(corpus.ids),segment_count=len(segments),split_passages=split,segments=segments,boundaries=boundaries,
        total_prefixed_segment_tokens=token_totals,
        budget_by_model={name:dict(tokens=specs[name]['corpus'].token_limit,characters=specs[name]['corpus'].character_limit,prefix=specs[name]['corpus'].prefix) for name in NAMES},
        cohere_tokenizer_hashes=tokenizer_hashes,local_artifact_fingerprints={name:a.fingerprint for name,a in artifacts.items()},
        tokenization_libraries={package:__import__('importlib.metadata',fromlist=['version']).version(package) for package in ('transformers','tokenizers','model2vec')},
        policy='Identical lossless slices for all accessible models; no overlap; original-text whitespace retained in boundaries; whitespace-only slices not embedded. Max segment similarity ranks unique parent passage IDs. No vector pooling or query chunking.')
    checked_save(path,catalog)
    event(directory,dict(stage='shared-layout-complete',passages=catalog['passage_count'],segments=catalog['segment_count'],split_passages=split,token_totals=token_totals))
    return catalog


def materialize(corpus,catalog):
    texts=tuple(corpus.texts[i][a:b] for i,a,b in catalog['segments'])
    owners=np.array([i for i,_,_ in catalog['segments']],dtype=np.int64)
    return texts,owners


class SingleSegmentEncoder:
    def __init__(self,base,specs,remote=False):
        self.base,self.specs,self.remote=base,specs,remote
        self.requests=[]
        self.dimension=base.model['dimension'] if remote else base.dimension

    def encode(self,texts,role):
        p=self.specs[role]
        if any(not p.fits(text) for text in texts):
            raise ValueError('Input violates frozen shared-segment budget')
        if self.remote:
            blocks=[]
            for a in range(0,len(texts),96):
                vectors,request=self.base.encode(texts[a:a+96],'search_document' if role=='corpus' else 'search_query')
                if request['character_truncated_inputs']:
                    raise AssertionError('Unexpected input truncation')
                self.requests.append(request); blocks.append(vectors)
            return np.concatenate(blocks) if blocks else np.empty((0,self.dimension),dtype=np.float32)
        if hasattr(self.base,'query_prefix'):
            result=self.base.model.encode([p.prefix+t for t in texts],max_length=None,batch_size=self.base.batch_size,use_multiprocessing=False)
        else:
            result=(self.base.encode_passages if role=='corpus' else self.base.encode_queries)(texts)
        return np.asarray(result,dtype=np.float32)


def measurement(name,encoder,corpus,catalog,prepared,vectors,owners,directory,provenance,artifact=None):
    path=directory/name/'measurement.json'
    if path.exists(): return verified_json(path,provenance)
    lookup={q.query_id:q.query for q in prepared.development}
    ids=catalog['development_query_ids']
    encodings,searches,total=[],[],[]
    request_start=len(encoder.requests)
    for repetition in range(7):
        for query_id in ids:
            before=time.perf_counter()
            query=encoder.encode([lookup[query_id]],'development')
            middle=time.perf_counter()
            parent_top_k(query,vectors,owners,corpus.ids,k=10)
            after=time.perf_counter()
            if repetition>=2:
                encodings.append(middle-before); searches.append(after-middle); total.append(after-before)
        event(directory,dict(stage='operational-measurement',model=name,round=repetition+1,rounds=7))
    result=dict(provenance=provenance,environment=host_metadata(),development_query_ids=ids,
        warm=dict(query_count=len(ids),warmup_rounds=2,repeated_rounds=5,single_query_encoding=_timing_summary(encodings),
                  single_query_exact_search=_timing_summary(searches),single_query_end_to_end=_timing_summary(total)),
        model_artifact_size_bytes=sum((artifact.path/f).stat().st_size for f in artifact.files_sha256) if artifact else None,
        peak_rss_after_workload_bytes=peak_rss_bytes() if artifact else None,peak_rss_client_bytes=peak_rss_bytes(),
        requests=encoder.requests[request_start:],
        note='Isolated process peak RSS includes corpus, segment metadata, mapped vectors, tokenizer and runtime; not weights-only RAM. Cohere server RAM/model artifact size unknown. Query encoding includes validation and network for Cohere; parent-max aggregation included in search latency.')
    checked_save(path,result)
    return result


def evaluate(name,encoder,corpus,catalog,queries,prepared,vectors,owners,directory,provenance,artifact=None):
    target=directory/name/'query-vectors.npy'
    if target.exists():
        query_vectors=publish_matrix(target,np.empty((0,0)),provenance)
    else:
        query_vectors=publish_matrix(target,encoder.encode([q.query for q in queries],'queries'),provenance)
        checked_save(directory/name/'query-encoding.json',dict(provenance=provenance,requests=encoder.requests))
    measured=measurement(name,encoder,corpus,catalog,prepared,vectors,owners,directory,provenance,artifact)
    rows=[]
    for a in range(0,len(queries),16):
        batch=queries[a:a+16]
        rankings=parent_top_k(query_vectors[a:a+len(batch)],vectors,owners,corpus.ids,k=10)
        for query,ranking in zip(batch,rankings):
            rows.append(dict(query_id=query.query_id,category=query.category,raw_grade3_count=query.raw_grade3_count,mapped_grade3_count=len(query.gold_ids),
                metrics=score_top10_query(query,ranking),top10=[dict(corpus_id=r.corpus_id,similarity=r.similarity) for r in ranking]))
        if (a//16+1)%10==0 or len(rows)==len(queries):
            event(directory,dict(stage='quality-scoring',model=name,completed_queries=len(rows),total_queries=len(queries)))
    result=dict(provenance=provenance,model=name,per_query=rows,measurement=measured,
        segment_vector_size_bytes=(directory/name/'segment-vectors.npy').stat().st_size,
        passage_count=len(corpus.ids),segment_count=catalog['segment_count'])
    if name=='cohere_v3':
        requests=read(directory/name/'corpus-encoding.json')['requests']+read(directory/name/'query-encoding.json')['requests']+measured['requests']
        if any(r['character_truncated_inputs'] for r in requests): raise AssertionError('Truncation detected')
        result['operational']=dict(input_tokens=sum(r['input_tokens'] or 0 for r in requests),requests_without_token_counts=sum(r['input_tokens'] is None for r in requests),
            estimated_api_cost_usd=sum(r['input_tokens'] or 0 for r in requests)/1e6*.1,
            successful_new_requests=len(requests),note='New invocations only; historical reused-vector spend excluded; credits unverified.')
    checked_save(directory/name/'query-results.json',result)
    event(directory,dict(stage='model-complete',model=name,metrics=summarize(rows,metrics=TOP10_METRICS)['overall']))


def local(name,config,directory,provenance):
    corpus,queries,prepared,*_=prepare(config)
    artifacts=local_model_artifacts(ROOT)
    artifact=artifacts[name]
    tokenizer_path,_=tokenizer_snapshot()
    catalog=verified_json(directory/'shared-catalog.json',provenance)
    texts,owners=materialize(corpus,catalog)
    base=artifact.load_encoder()
    encoder=SingleSegmentEncoder(base,policies(name,artifacts,tokenizer_path))
    target=directory/name/'segment-vectors.npy'
    if target.exists():
        vectors=publish_matrix(target,np.empty((0,0)),provenance)
    else:
        source=ROOT/'runs'/'embeddings'/f'{name}-{artifact.fingerprint[:12]}.npy'
        old=open_corpus_embeddings(source,corpus=corpus,corpus_sha256=config['corpus_sha256'],model_fingerprint=artifact.fingerprint)
        values=np.zeros((len(texts),encoder.dimension),dtype=np.float32)
        changed=[]
        reused=0
        for row,(owner,a,b) in enumerate(catalog['segments']):
            if a==0 and b==len(corpus.texts[owner]):
                values[row]=old[owner]; reused+=1
            else: changed.append(row)
        for a in range(0,len(changed),64):
            selected=changed[a:a+64]
            values[selected]=encoder.encode([texts[i] for i in selected],'corpus')
            event(directory,dict(stage='local-segment-encoding',model=name,completed_segments=min(a+64,len(changed)),new_segments=len(changed),reused_complete_passages=reused))
        vectors=publish_matrix(target,values,provenance)
        checked_save(directory/name/'reuse.json',dict(provenance=provenance,source=str(source.relative_to(ROOT)),source_sha256=file_digest(source),
            source_fingerprint=artifact.fingerprint,reused_complete_passages=reused,new_segment_rows=changed))
        del old,values
        gc.collect()
    evaluate(name,encoder,corpus,catalog,queries,prepared,vectors,owners,directory,provenance,artifact)


def cohere(config,directory,provenance,corpus,queries,prepared,artifacts,tokenizer_path,catalog):
    import boto3
    from botocore.config import Config
    client=boto3.Session(profile_name=config['aws_profile']).client('bedrock-runtime',region_name=config['aws_region'],
        config=Config(connect_timeout=10,read_timeout=90,max_pool_connections=6,retries={'max_attempts':0}))
    settings={**config['models']['cohere_v3'],'truncate':'NONE','maximum_input_characters':2048,'full_content':True}
    base=BedrockCohere(client,settings,tokens_per_minute=config['tokens_per_minute'])
    encoder=SingleSegmentEncoder(base,policies('cohere_v3',artifacts,tokenizer_path),remote=True)
    texts,owners=materialize(corpus,catalog)
    target=directory/'cohere_v3'/'segment-vectors.npy'
    if target.exists():
        vectors=publish_matrix(target,np.empty((0,0)),provenance)
    else:
        reusable={}
        for meta_path in sorted((ROOT/'runs/cohere-grade3-3ac887ab565e/cohere_v3/search_document').glob('*.json')):
            meta=read(meta_path); p=meta['provenance']; source=meta_path.with_suffix('.npy')
            a,b=p['start'],p['stop']
            if any(p[key]!=expected for key,expected in [('corpus_sha256',config['corpus_sha256']),('role','search_document'),('model','cohere_v3'),('aws_profile',config['aws_profile']),('aws_region',config['aws_region'])]):
                raise ValueError('Historical Cohere provenance mismatch')
            if p['texts_sha256']!=digest(corpus.texts[a:b]) or p['settings']['model_id']!=settings['model_id'] or p['settings']['dimension']!=1024:
                raise ValueError('Historical Cohere input/model mismatch')
            if not source.is_file() or file_digest(source)!=meta['vectors_sha256']:
                raise ValueError('Historical Cohere batch checksum mismatch')
            old=np.load(source,mmap_mode='r')
            if old.shape!=(b-a,1024) or not np.isfinite(old).all() or not np.allclose(np.linalg.norm(old,axis=1),1,atol=2e-5):
                raise ValueError('Historical vectors invalid')
            for owner in range(a,b):
                if len(catalog['boundaries'][owner])==1:
                    reusable[owner]=(source,owner-a,meta['vectors_sha256'])
        values=np.zeros((len(texts),1024),dtype=np.float32)
        missing=[]
        source_groups={}
        reuse_proof=[]
        for row,(owner,a,b) in enumerate(catalog['segments']):
            if owner in reusable and a==0 and b==len(corpus.texts[owner]):
                source,j,checksum=reusable[owner]
                source_groups.setdefault(source,[]).append((row,j))
                reuse_proof.append(dict(segment_row=row,parent_row=owner,source=str(source.relative_to(ROOT)),source_row=j,source_sha256=checksum))
            else: missing.append(row)
        for source,selection in source_groups.items():
            old=np.load(source,mmap_mode='r')
            values[[i for i,_ in selection]]=old[[j for _,j in selection]]
        checked_save(directory/'cohere_v3'/'reuse.json',dict(provenance=provenance,reused_complete_passages=len(reuse_proof),proof=reuse_proof,new_segment_rows=missing))
        pending=tuple(texts[i] for i in missing)
        new_config={key:value for key,value in config.items() if key!='reuse_verified_batches_from'}
        metas,paths=encode_batches('cohere_v3','search_document',pending,base,new_config,directory,
            {**provenance,'model':'cohere_v3','settings':settings,'query_ids_sha256':digest([q.query_id for q in queries]),'aws_profile':config['aws_profile'],'aws_region':config['aws_region'],'segment_rows_sha256':digest(missing)})
        for meta,path in zip(metas,paths):
            p=meta['provenance']; batch=np.load(path,mmap_mode='r')
            if meta['request']['character_truncated_inputs']: raise AssertionError('Unexpected truncation')
            values[missing[p['start']:p['stop']]]=batch
        vectors=publish_matrix(target,values,provenance)
        checked_save(directory/'cohere_v3'/'corpus-encoding.json',dict(provenance=provenance,requests=[m['request'] for m in metas],
            new_segments=len(missing),reused_complete_passages=len(reuse_proof)))
        del values
        gc.collect()
    evaluate('cohere_v3',encoder,corpus,catalog,queries,prepared,vectors,owners,directory,provenance)


def protocol_and_directory():
    config=read(ROOT/'configs/cohere-grade3-v2.json')
    files=['scripts/run_parent_segment_grade3.py','scripts/run_full_content_grade3.py','scripts/run_cohere_grade3.py',
        'src/pycon_model2vec_demo/parent_retrieval.py','src/pycon_model2vec_demo/bedrock_embeddings.py','src/pycon_model2vec_demo/segmented_encoding.py']
    protocol=dict(base_config=config,method='shared-segment-parent-max-v1',query_policy='Unchanged queries; validate against all model budgets; fail if oversized',
        corpus_policy='Identical lossless nonoverlapping slices satisfying all accessible model budgets; no vector pooling',
        retrieval='Exact all-segment similarity, maximum score per original passage, lexical ties, top ten unique original passage IDs',
        cohere_tokenizer=dict(repo=TOKENIZER_REPO,revision=TOKENIZER_REVISION),cohere_truncate='NONE',cohere_safe_tokens=480,
        source_hashes={file:file_digest(ROOT/file) for file in files})
    sha=digest(protocol)
    directory=ROOT/'runs'/('parent-segment-grade3-'+sha[:12])
    return config,protocol,directory,dict(protocol_sha256=sha,corpus_sha256=config['corpus_sha256'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local',choices=MODEL_NAMES)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    config,protocol,directory,provenance=protocol_and_directory()
    directory.mkdir(parents=True,exist_ok=True)
    save(directory/'protocol.json',protocol)
    if args.local:
        local(args.local,config,directory,provenance)
        return
    lock=(directory/'runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    event(directory,dict(stage='preparing-parent-segment-run',pid=os.getpid()))
    tokenizer_path,hashes=tokenizer_snapshot()
    artifacts=local_model_artifacts(ROOT)
    corpus,queries,prepared,_,selection,reference=prepare(config)
    catalog=make_catalog(corpus,queries,prepared,artifacts,tokenizer_path,hashes,directory,provenance)
    if args.preflight_only: return
    for name in MODEL_NAMES:
        if not (directory/name/'query-results.json').exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--local',name],check=True,cwd=ROOT,env=os.environ.copy())
    if not (directory/'cohere_v3'/'query-results.json').exists():
        cohere(config,directory,provenance,corpus,queries,prepared,artifacts,tokenizer_path,catalog)
    models,rows={},{}
    for name in NAMES:
        result=verified_json(directory/name/'query-results.json',provenance)
        rows[name]=result['per_query']
        models[name]=dict(status='completed',quality=summarize(rows[name],metrics=TOP10_METRICS),measurement=result['measurement'],
            corpus_vector_size_bytes=result['segment_vector_size_bytes'])
        if 'operational' in result: models[name]['operational']=result['operational']
    models['cohere_v4']=dict(status='blocked',error='Previously verified AWS Marketplace INVALID_PAYMENT_INSTRUMENT; no additional invocation attempted, no substitute.')
    report=dict(provenance=provenance,status='completed-with-blocked-models',query_count=len(queries),corpus_passage_count=len(corpus.ids),segment_count=catalog['segment_count'],
        corpus_sha256=config['corpus_sha256'],protocol_sha256=provenance['protocol_sha256'],top_k=10,selection_audit=selection,
        reference_cohort_sha256=reference['provenance']['cohort_sha256'],query_ids=[q.query_id for q in queries],models=models,
        paired_with_teacher={name:paired_with_teacher(rows['teacher'],r,metrics=TOP10_METRICS) for name,r in rows.items() if name!='teacher'},
        method=protocol['method'],shared_catalog_sha256=file_digest(directory/'shared-catalog.json'))
    path=directory/'reports'/'parent-segment-report.json'
    checked_save(path,report)
    text=render(report).replace('Corpus vectors MiB','Segment vectors MiB')+'\nAll models use the same '+str(catalog['segment_count'])+' independently searchable segments. Each original passage is ranked by its best segment; ten means ten unique original passage IDs. This measures parent-passage retrieval, not answer-bearing segment accuracy. There is no vector averaging. RSS is whole isolated-process memory, not weights-only RAM. Cohere invocation cost excludes historical reused-vector spend; AWS credits are not confirmed.\n'
    md=path.with_suffix('.md')
    if md.exists():
        if md.read_text()!=text: raise ValueError('Existing report differs')
    else:
        with md.open('x') as output: output.write(text)
    event(directory,dict(stage='report-complete',report=str(path.relative_to(ROOT)),status=report['status']))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        _,_,directory,_=protocol_and_directory()
        event(directory,dict(stage='run-failed',error_type=type(exc).__name__,message=str(exc)))
        raise
