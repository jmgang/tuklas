"""Controlled development-only replay of the approved student's ranking regression.

Only learning_rate varies. The executed ranking helper, train pairs, batches,
loss, seed, 1024 dimensions, teacher-derived weights and validator are frozen.
No evaluation-partition question is scored or used for checkpoint selection.
"""
import os
from pathlib import Path
import sys
import json
from hashlib import sha256
import importlib
from collections import Counter

INPUT = Path('/opt/ml/input/data/input')
sys.path.insert(0, str(INPUT))
os.environ['SEALION_INPUT_DIR'] = str(INPUT)
os.environ['SEALION_WORK_DIR'] = '/opt/ml/lr-probe-work'
original = importlib.import_module('aws_sealion_train_repair_v3')
PLAN = json.loads((INPUT/'probe-plan.json').read_text())


def fetch(preserve, spec, destination):
    h = sha256()
    response = preserve.s3.get_object(Bucket=PLAN['bucket'], Key=spec['key'])
    with destination.open('xb') as target:
        for block in iter(lambda: response['Body'].read(2**20), b''):
            h.update(block)
            target.write(block)
    response['Body'].close()
    if destination.stat().st_size != spec['bytes'] or h.hexdigest() != spec['sha256']:
        raise ValueError('Frozen input checksum changed: '+spec['key'])


def main():
    original.WORK.mkdir(exist_ok=False, parents=True)
    (original.WORK/'events').mkdir()
    preserve = original.Preservation()
    for name, spec in PLAN['source_files'].items():
        path = INPUT/name
        if path.stat().st_size != spec['bytes'] or original.digest(path) != spec['sha256']:
            raise ValueError('Staged source changed: '+name)
    preserve.event('lr-probe-dependency-setup')
    original.bootstrap()
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from safetensors import safe_open
    from multiwikiqa_evaluation import rank_parents, score, summarize
    from multiwikiqa_ranking import prepare, train_ranked
    if not torch.cuda.is_available():
        raise RuntimeError('Expected SageMaker GPU')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    base = original.WORK/'approved-articles-base'
    base.mkdir(exist_ok=False)
    for filename, spec in PLAN['frozen_student'].items():
        fetch(preserve, spec, base/filename)
    preserve.event('lr-probe-frozen-student-restored', model_sha256=original.digest(base/'model.safetensors'))
    for name in ('annotations', 'evaluation_layout'):
        fetch(preserve, PLAN[name], original.WORK/(name+'.json'))
    queries = [json.loads(line) for line in (original.WORK/'annotations.json').read_text().splitlines()]
    layout = json.loads((original.WORK/'evaluation_layout.json').read_text())
    chunks = layout['chunks']
    if (len(queries),len(chunks),len(layout['parents'])) != (10000,6546,1294):
        raise ValueError('Evaluation layout or questions changed')
    train, validation, skipped = prepare(queries,chunks)
    if (len(train),len(validation),len(skipped)) != (1734,338,6):
        raise ValueError('Development split changed')
    if len({r['parent_id'] for r in train}) != 222:
        raise ValueError('Distinct training article count changed')
    positions = {name:i for i,name in enumerate(layout['parents'])}
    owners = [positions[c['parent_id']] for c in chunks]
    def evaluate(model, cohort):
        if any(q['partition']!='development' for q in cohort):
            raise ValueError('Held-out evaluation is forbidden in LR probe')
        documents = model.encode([c['text'] for c in chunks],batch_size=512,max_length=None)
        questions = model.encode([q['question'] for q in cohort],batch_size=512,max_length=None)
        results=[]
        for start in range(0,len(cohort),64):
            rankings=rank_parents(questions[start:start+64],documents,owners,layout['parents'])
            for q,ranking in zip(cohort[start:start+64],rankings):
                results.append({k:q[k] for k in ('query_id','gold_parent_id','language','partition','exclusion_reasons')}
                    | {'metrics':score(q,ranking,chunks), 'top10':[{'parent_id':parent,
                        'chunk_id':chunks[index]['chunk_id'],'cosine':cosine}
                        for parent,index,cosine in ranking]})
        return {'question_count':len(cohort),'per_query':results,'summary':summarize(results),
                'vector_bytes':documents.nbytes,'query_vector_bytes':questions.nbytes}
    tokenizer=Tokenizer.from_file(str(base/'tokenizer.json'))
    frequent=Counter(i for row in train for text in (row['question'],row['positive'])
                     for i in tokenizer.encode(text,add_special_tokens=False).ids)
    top_ids=[i for i,_ in frequent.most_common(40)]
    def geometry(before,after):
        with safe_open(str(before/'model.safetensors'),framework='numpy') as a:
            x=a.get_tensor('embeddings')[top_ids]
        with safe_open(str(after/'model.safetensors'),framework='numpy') as b:
            y=b.get_tensor('embeddings')[top_ids]
        nx=np.linalg.norm(x,axis=1);ny=np.linalg.norm(y,axis=1)
        cosine=np.sum(x*y,axis=1)/(nx*ny+1e-12)
        relative=np.linalg.norm(y-x,axis=1)/(nx+1e-12)
        return {'frequent_token_count':40,'median_row_cosine':float(np.median(cosine)),
                'median_relative_l2_change':float(np.median(relative)),
                'rows_relative_change_above_one':int(np.sum(relative>1)),
                'ng_row_cosine':float(cosine[top_ids.index(232)]) if 232 in top_ids else None}
    results=[]
    for label, lr in PLAN['learning_rates']:
        recipe={**PLAN['ranking'],'epochs':1,'learning_rate':lr}
        name='approved_articles_'+label
        selected,report=train_ranked(name,base,queries,chunks,evaluate,preserve,original.WORK,recipe)
        directory=original.WORK/(name+'-ranking')
        v=[]
        for epoch in (0,1):
            data=json.loads((directory/f'validation-{epoch:02d}.json').read_text())
            summary={}
            for language in ('en','tl'):
                rows=[r for r in data['per_query'] if r['language']==language]
                summary[language]={'questions':len(rows),
                    'nDCG@10':float(np.mean([r['metrics']['nDCG@10'] for r in rows])),
                    'Recall@10':float(np.mean([r['metrics']['Recall@10'] for r in rows]))}
            v.append({'epoch':epoch,'combined_nDCG@10':report['history'][epoch]['nDCG@10'],
                      'combined_Recall@10':data['summary']['development_partition']['metrics']['Recall@10']['mean'],
                      'by_language':summary})
        outcome={'label':label,'learning_rate':lr,'validation':v,
                 'token_geometry':geometry(directory/'epoch-00',directory/'epoch-01'),
                 'selected_epoch':report['selected_epoch'],
                 'actual_batches':report['history'][1]['batch_sizes']}
        original.write_json(original.WORK/(label+'-outcome.json'),outcome)
        preserve.put(original.WORK/(label+'-outcome.json'),'reports/'+label+'-outcome.json')
        preserve.event('lr-probe-candidate-complete',label=label,learning_rate=lr,
                       dev_ndcg_before=v[0]['combined_nDCG@10'],
                       dev_ndcg_after=v[1]['combined_nDCG@10'])
        results.append(outcome)
    control=results[0]
    if control['label']!='control_0p05' or abs(control['validation'][1]['combined_nDCG@10']-0.3597142812932725)>0.02:
        raise ValueError('Control failed to reproduce original regression; no causal conclusion')
    final={'status':'complete','hypothesis':'Reduce only learning rate from .05',
           'same_model_data_objective_batches_seed_and_evaluation':True,
           'heldout_scored':False,'frozen_model_hash':original.digest(base/'model.safetensors'),
           'development_train_pairs':len(train),'development_train_articles':222,
           'development_validation_questions':len(validation),'results':results,
           'source_plan_sha256':original.digest(INPUT/'probe-plan.json'),'artifacts':preserve.index}
    original.write_json(original.WORK/'lr-probe-completion.json',final)
    preserve.put(original.WORK/'lr-probe-completion.json','reports/lr-probe-completion.json')
    preserve.event('lr-probe-complete',heldout_scored=False,candidates=len(results))


if __name__=='__main__':
    main()
