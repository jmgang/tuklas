"""Exact first-epoch LR replay: original three-epoch schedule, no held-out scores."""
import gc
import importlib
import json
import os
from pathlib import Path
import sys
from collections import Counter
from hashlib import sha256

INPUT=Path('/opt/ml/input/data/input')
sys.path.insert(0,str(INPUT))
os.environ['SEALION_INPUT_DIR']=str(INPUT)
os.environ['SEALION_WORK_DIR']='/opt/ml/lr-probe-v2-work'
original=importlib.import_module('aws_sealion_train_repair_v3')
PLAN=json.loads((INPUT/'probe-plan.json').read_text())


def fetch(preserve,spec,destination):
    response=preserve.s3.get_object(Bucket=PLAN['bucket'],Key=spec['key'])
    h=sha256()
    with destination.open('xb') as target:
        for block in iter(lambda:response['Body'].read(2**20),b''):
            h.update(block);target.write(block)
    response['Body'].close()
    if destination.stat().st_size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:
        raise ValueError('Frozen input bytes changed: '+spec['key'])


class EndOfFirstEpoch(Exception):
    pass


class FirstEpochPreservation:
    """Save both checkpoints, then stop AFTER the real original epoch-1 event."""
    def __init__(self,preserve):
        self.preserve=preserve
        self.epoch_one_event=None
    def put(self,*args,**kwargs):
        return self.preserve.put(*args,**kwargs)
    def tree(self,*args,**kwargs):
        return self.preserve.tree(*args,**kwargs)
    def event(self,stage,**detail):
        self.preserve.event(stage,**detail)
        if stage=='ranking-epoch-complete' and detail['epoch']==1:
            self.epoch_one_event=detail
            raise EndOfFirstEpoch


def main():
    original.WORK.mkdir(parents=True,exist_ok=False)
    (original.WORK/'events').mkdir()
    preserve=original.Preservation()
    for name,spec in PLAN['source_files'].items():
        path=INPUT/name
        if path.stat().st_size!=spec['bytes'] or original.digest(path)!=spec['sha256']:
            raise ValueError('Staged source changed: '+name)
    preserve.event('lr-probe-v2-dependency-setup')
    original.bootstrap()
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from safetensors import safe_open
    from multiwikiqa_evaluation import rank_parents,score,summarize
    from multiwikiqa_ranking import prepare,train_ranked
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    base=original.WORK/'approved-articles-base'
    base.mkdir(exist_ok=False)
    for filename,spec in PLAN['frozen_student'].items():
        fetch(preserve,spec,base/filename)
    preserve.event('frozen-student-restored',weights_sha256=original.digest(base/'model.safetensors'))
    for name in ('annotations','evaluation_layout'):
        fetch(preserve,PLAN[name],original.WORK/(name+'.json'))
    queries=[json.loads(line) for line in (original.WORK/'annotations.json').read_text().splitlines()]
    layout=json.loads((original.WORK/'evaluation_layout.json').read_text())
    chunks=layout['chunks']
    if (len(queries),len(chunks),len(layout['parents']))!=(10000,6546,1294):
        raise ValueError('Frozen data changed')
    train,validation,skipped=prepare(queries,chunks)
    if (len(train),len(validation),len(skipped))!=(1734,338,6):
        raise ValueError('Development split changed')
    if len({r['parent_id'] for r in train})!=222:
        raise ValueError('Distinct training article count changed')
    positions={name:i for i,name in enumerate(layout['parents'])}
    owners=[positions[c['parent_id']] for c in chunks]
    def evaluate(model,cohort):
        if any(q['partition']!='development' for q in cohort):
            raise ValueError('Evaluation partition forbidden in LR probe')
        documents=model.encode([c['text'] for c in chunks],batch_size=512,max_length=None)
        questions=model.encode([q['question'] for q in cohort],batch_size=512,max_length=None)
        results=[]
        for start in range(0,len(cohort),64):
            rankings=rank_parents(questions[start:start+64],documents,owners,layout['parents'])
            for q,ranking in zip(cohort[start:start+64],rankings):
                results.append({k:q[k] for k in ('query_id','gold_parent_id','language','partition','exclusion_reasons')}
                    | {'metrics':score(q,ranking,chunks),'top10':[{'parent_id':parent,
                        'chunk_id':chunks[index]['chunk_id'],'cosine':cosine}
                        for parent,index,cosine in ranking]})
        return {'question_count':len(cohort),'per_query':results,'summary':summarize(results),
                'vector_bytes':documents.nbytes,'query_vector_bytes':questions.nbytes}
    tokenizer=Tokenizer.from_file(str(base/'tokenizer.json'))
    frequent=Counter(i for row in train for value in (row['question'],row['positive'])
                     for i in tokenizer.encode(value,add_special_tokens=False).ids)
    top_ids=[i for i,_ in frequent.most_common(40)]
    def geometry(before,after):
        with safe_open(str(before/'model.safetensors'),framework='numpy') as file:
            x=file.get_tensor('embeddings')[top_ids]
        with safe_open(str(after/'model.safetensors'),framework='numpy') as file:
            y=file.get_tensor('embeddings')[top_ids]
        nx=np.linalg.norm(x,axis=1);ny=np.linalg.norm(y,axis=1)
        cosine=np.sum(x*y,axis=1)/(nx*ny+1e-12)
        relative=np.linalg.norm(y-x,axis=1)/(nx+1e-12)
        return {'frequent_token_count':40,'median_row_cosine':float(np.median(cosine)),
                'median_relative_l2_change':float(np.median(relative)),
                'rows_relative_change_above_one':int(np.sum(relative>1)),
                'ng_row_cosine':float(cosine[top_ids.index(232)]) if 232 in top_ids else None}
    outcomes=[]
    for label,lr in PLAN['learning_rates']:
        recipe={**PLAN['ranking'],'learning_rate':lr}
        if recipe['epochs']!=3:
            raise ValueError('Original three-epoch schedule must remain fixed')
        name='approved_articles_'+label
        wrapper=FirstEpochPreservation(preserve)
        try:
            train_ranked(name,base,queries,chunks,evaluate,wrapper,original.WORK,recipe)
        except EndOfFirstEpoch:
            pass
        else:
            raise ValueError('Failed to stop exactly after first original-schedule epoch')
        if wrapper.epoch_one_event is None:
            raise ValueError('Missing first-epoch event')
        directory=original.WORK/(name+'-ranking')
        validation_results=[]
        for epoch in (0,1):
            data=json.loads((directory/f'validation-{epoch:02d}.json').read_text())
            by_language={}
            for language in ('en','tl'):
                rows=[r for r in data['per_query'] if r['language']==language]
                by_language[language]={'questions':len(rows),
                    'nDCG@10':float(np.mean([r['metrics']['nDCG@10'] for r in rows])),
                    'Recall@10':float(np.mean([r['metrics']['Recall@10'] for r in rows]))}
            validation_results.append({'epoch':epoch,
                'nDCG@10':data['summary']['development_partition']['metrics']['nDCG@10']['mean'],
                'Recall@10':data['summary']['development_partition']['metrics']['Recall@10']['mean'],
                'by_language':by_language})
        outcome={'label':label,'learning_rate':lr,'validation':validation_results,
                 'token_geometry':geometry(directory/'epoch-00',directory/'epoch-01'),
                 'first_epoch_event':wrapper.epoch_one_event,
                 'original_three_epoch_schedule_preserved':True}
        original.write_json(original.WORK/(label+'-outcome.json'),outcome)
        preserve.put(original.WORK/(label+'-outcome.json'),'reports/'+label+'-outcome.json')
        preserve.event('lr-candidate-complete',label=label,
            nDCG_before=validation_results[0]['nDCG@10'],nDCG_after=validation_results[1]['nDCG@10'])
        outcomes.append(outcome)
        gc.collect();torch.cuda.empty_cache()
    control=outcomes[0]
    if control['label']!='control_0p05' or abs(control['validation'][1]['nDCG@10']-0.3597142812932725)>0.02:
        raise ValueError('Control did not reproduce old regression; intervention inconclusive')
    final={'status':'complete','tested_hypothesis':'Only learning rate changes',
           'same_model_data_seed_objective_batch_order_and_three_epoch_schedule':True,
           'heldout_scored':False,'initial_weights_sha256':original.digest(base/'model.safetensors'),
           'development_train_pairs':1734,'development_train_articles':222,
           'development_validation_questions':338,'outcomes':outcomes,
           'source_plan_sha256':original.digest(INPUT/'probe-plan.json'),
           'artifacts':preserve.index}
    original.write_json(original.WORK/'lr-probe-completion.json',final)
    preserve.put(original.WORK/'lr-probe-completion.json','reports/lr-probe-completion.json')
    preserve.event('lr-probe-complete',heldout_scored=False,candidates=len(outcomes))


if __name__=='__main__':
    main()
