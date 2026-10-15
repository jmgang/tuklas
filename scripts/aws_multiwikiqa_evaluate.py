"""Immutable as-released 10,000-question evaluation; never trains a model."""
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

INPUT=Path('/opt/ml/input/data/input')
OUT=Path('/opt/ml/evaluation-results')


def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def main():
    # Explicit dependencies needed by these actual inference paths, not TokenLearn training.
    subprocess.run([sys.executable,'-m','pip','install','model2vec==0.9.0',
        'sentence-transformers==5.7.0','transformers==5.3.0','numpy==2.5.3'],check=True)
    import boto3
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer
    from model2vec import StaticModel
    from multiwikiqa_evaluation import rank_parents,score,summarize,paired_cluster_ci,validate_questions
    torch.set_num_threads(4)
    settings=json.loads((INPUT/'settings.json').read_text())
    s3=boto3.client('s3',region_name='ap-southeast-1')
    bucket=os.environ['EVAL_BUCKET']
    prefix=os.environ['EVAL_PREFIX']
    OUT.mkdir(exist_ok=False)
    artifacts={}

    def publish(path):
        name=str(path.relative_to(OUT))
        spec={'bytes':path.stat().st_size,'sha256':digest(path)}
        with path.open('rb') as f:
            s3.put_object(Bucket=bucket,Key=prefix+'/'+name,Body=f,
                ContentLength=spec['bytes'],IfNoneMatch='*',Metadata={'sha256':spec['sha256']})
        artifacts[name]=spec
        return spec

    def save(name,value):
        path=OUT/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:
            json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2)
            f.write('\n')
        publish(path)

    def event(stage,**extra):
        value={'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**extra}
        save('events/'+str(time.time_ns())+'-'+uuid4().hex[:6]+'.json',value)
        print(json.dumps(value),flush=True)

    def rows(name):
        spec=settings['data_files'][name]
        if digest(INPUT/name)!=spec['sha256']:
            raise ValueError('Input hash changed: '+name)
        with (INPUT/name).open() as f:
            return [json.loads(line) for line in f]

    parents=rows('parents.jsonl')
    queries=rows('annotations.all.jsonl')
    candidates=rows('chunks.candidate.jsonl')
    validate_questions(queries,parents)
    if len(candidates)!=6261:
        raise ValueError('Original candidate chunks changed')
    save('settings.json',settings)
    event('inputs-verified',questions=len(queries),parents=len(parents),candidate_chunks=len(candidates))

    # Models are staged from existing immutable weights; full bytes checked in-region.
    paths={}
    for name,specs in settings['static_models'].items():
        path=OUT/'model-inputs'/name
        path.mkdir(parents=True,exist_ok=False)
        for filename,spec in specs.items():
            target=path/filename
            response=s3.get_object(Bucket=bucket,Key=spec['key'])
            with target.open('xb') as f:
                for block in iter(lambda:response['Body'].read(2**20),b''):
                    f.write(block)
            response['Body'].close()
            if target.stat().st_size!=spec['bytes'] or digest(target)!=spec['sha256']:
                raise ValueError('Frozen model checksum mismatch: '+name+'/'+filename)
        paths[name]=path
        event('frozen-static-model-verified',model=name)

    event('loading-teacher')
    teacher=SentenceTransformer(settings['teacher']['model_id'],
        revision=settings['teacher']['revision'],device='cuda',trust_remote_code=False)
    if not torch.cuda.is_available() or teacher.get_sentence_embedding_dimension()!=1024:
        raise ValueError('Expected GPU teacher and 1024 dimensions')
    query_prompt=settings['teacher']['query_prompt']
    if teacher.prompts.get('Retrieval')!=query_prompt:
        raise ValueError('Pinned teacher Retrieval prompt changed')
    teacher.max_seq_length=512
    potion_tokenizer=Tokenizer.from_file(str(paths['potion_retrieval']/'tokenizer.json'))
    potion_tokenizer.no_truncation()
    student_tokenizer=Tokenizer.from_file(str(paths['approved_articles']/'tokenizer.json'))
    student_tokenizer.no_truncation()

    def fits(text):
        return (len(teacher.tokenizer.encode(text,add_special_tokens=True))<=512
            and len(potion_tokenizer.encode(text,add_special_tokens=False).ids)<=512
            and len(student_tokenizer.encode(text,add_special_tokens=False).ids)<=512)

    parent_index={p['parent_id']:i for i,p in enumerate(parents)}
    chunks=[]
    for piece in candidates:
        start=0
        while start<len(piece['text']):
            stop=len(piece['text'])
            while stop>start and not fits(piece['text'][start:stop]):
                stop-=1
            if stop==start:
                raise ValueError('Single source character cannot fit frozen budgets')
            a,b=piece['start']+start,piece['start']+stop
            if parents[parent_index[piece['parent_id']]]['text'][a:b]!=piece['text'][start:stop]:
                raise ValueError('Original source offsets changed')
            chunks.append({'chunk_id':piece['parent_id']+f':{a}:{b}',
                'parent_id':piece['parent_id'],'language':piece['language'],
                'start':a,'end':b,'text':piece['text'][start:stop]})
            start=stop
    chunks.sort(key=lambda r:(parent_index[r['parent_id']],r['start'],r['end']))
    for q in queries:
        text=q['question']
        if (len(teacher.tokenizer.encode(query_prompt+text,add_special_tokens=True))>512
                or len(potion_tokenizer.encode(text,add_special_tokens=False).ids)>512
                or len(student_tokenizer.encode(text,add_special_tokens=False).ids)>512):
            raise ValueError('Over-budget query: '+q['query_id']+'; no silent truncation')
    owners=[parent_index[c['parent_id']] for c in chunks]
    ids=[p['parent_id'] for p in parents]
    save('shared-layout.json',{'parents':ids,'chunks':chunks,'query_ids':[q['query_id'] for q in queries],
        'parent_reduction':'maximum chunk cosine','cutoff':10,'query_filtering':False,
        'semantic_review':'pending/as-released','query_gold':'one unchanged known source article'})
    layout_hash=artifacts['shared-layout.json']['sha256']
    event('shared-layout-frozen',segments=len(chunks),questions=10000,layout_sha256=layout_hash)

    def encode(model,texts,is_teacher,is_query):
        batches=[]
        size=32 if is_teacher else 512
        for start in range(0,len(texts),size):
            batch=texts[start:start+size]
            if is_teacher:
                v=model.encode(batch,batch_size=size,normalize_embeddings=True,
                    convert_to_numpy=True,prompt=query_prompt if is_query else '',show_progress_bar=False)
            else:
                v=model.encode(batch,batch_size=size,max_length=None,use_multiprocessing=False)
            v=np.asarray(v,dtype=np.float32)
            if v.ndim!=2 or len(v)!=len(batch) or not np.isfinite(v).all():
                raise ValueError('Bad encoded batch')
            batches.append(v)
            if (start//size)%20==0 or start+len(batch)==len(texts):
                event('encoding',model=current,kind='queries' if is_query else 'chunks',
                      rows=start+len(batch),total=len(texts))
        return np.concatenate(batches)

    reports={}
    for current in ('teacher','approved_articles','external_wikipedia','potion_retrieval'):
        event('model-evaluation-start',model=current)
        is_teacher=current=='teacher'
        model=teacher if is_teacher else StaticModel.from_pretrained(str(paths[current]))
        documents=encode(model,[c['text'] for c in chunks],is_teacher,False)
        questions=encode(model,[q['question'] for q in queries],is_teacher,True)
        for kind,matrix in [('chunks',documents),('queries',questions)]:
            file=OUT/'vectors'/current/(kind+'.npy')
            file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('xb') as f:
                np.save(f,matrix,allow_pickle=False)
            publish(file)
        results=[]
        for start in range(0,len(queries),64):
            rankings=rank_parents(questions[start:start+64],documents,owners,ids)
            for q,ranking in zip(queries[start:start+64],rankings):
                results.append({k:q[k] for k in ('query_id','gold_parent_id','language','partition','exclusion_reasons')}
                    | {'metrics':score(q,ranking,chunks),
                       'top10':[{'parent_id':p,'chunk_id':chunks[i]['chunk_id'],'cosine':s} for p,i,s in ranking]})
        weight_bytes=(sum(s['bytes'] for f,s in settings['static_models'][current].items()
                            if f.endswith('safetensors')) if not is_teacher else
                            sum(p.numel()*p.element_size() for p in model.parameters()))
        report={'model':current,'question_count':len(results),'shared_layout_sha256':layout_hash,
            'summary':summarize(results),'per_query':results,'model_weights_bytes':weight_bytes,
            'vector_bytes':documents.nbytes,'query_vector_bytes':questions.nbytes,
            'zero_query_vectors':int((np.linalg.norm(questions,axis=1)==0).sum()),
            'zero_chunk_vectors':int((np.linalg.norm(documents,axis=1)==0).sum()),
            'latency_measured':False,'runtime_memory_measured':False}
        save('reports/'+current+'.json',report)
        reports[current]=report
        event('model-evaluation-complete',model=current,questions=len(results),
              metrics=report['summary']['all_10000']['metrics'])
        if is_teacher:
            teacher=None
        del model,documents,questions
        gc.collect()
        torch.cuda.empty_cache()

    paired={}
    for a,b in [('approved_articles','potion_retrieval'),('external_wikipedia','potion_retrieval'),
                ('approved_articles','external_wikipedia')]:
        paired[a+' minus '+b]={m:paired_cluster_ci(reports[a]['per_query'],reports[b]['per_query'],m)
                              for m in ('Recall@10','nDCG@10','MRR@10')}
    combined={'status':'complete','models':{n:{k:v for k,v in r.items() if k!='per_query'} for n,r in reports.items()},
        'paired_article_cluster_intervals':paired,'dataset':settings['dataset'],
        'questions':10000,'parents':1294,'segments':len(chunks),'cutoff':10,
        'semantic_review':'pending/as-released; not human validated',
        'empty_answer_references':10,'coverage_denominator':9990,
        'answer_generation_evaluated':False,'latency_measured':False,
        'limitations':['One known source parent, not exhaustive relevance',
            'Approved-article student has seen corpus text, not questions/answers',
            'Training corpora differ in size; validation loss is not retrieval quality',
            'All-question score includes previously designated development rows',
            'Synthetic questions and recorded annotations have unresolved semantic issues'],
        'settings_sha256':artifacts['settings.json']['sha256'],'shared_layout_sha256':layout_hash}
    save('reports/combined.json',combined)
    save('reports/completion.json',{'status':'complete','artifacts':dict(artifacts),
        'models':list(reports),'questions':10000,'compute_release':'managed job releases on exit'})
    event('complete',models=4,questions=10000)


if __name__=='__main__':
    main()
