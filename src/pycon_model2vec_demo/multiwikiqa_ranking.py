"""Development-only ranking pairs and duplicate-safe in-batch negatives."""
from hashlib import sha256
import unicodedata


def key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prepare(queries, chunks):
    forbidden = {q['query_id'] for q in queries if q['partition']=='evaluation'}
    forbidden_text = {key(q['question']) for q in queries if q['partition']=='evaluation'}
    by_parent = {}
    for chunk in chunks:
        by_parent.setdefault(chunk['parent_id'], []).append(chunk)
    references = {}
    for q in queries:
        references.setdefault(key(q['question']),set()).add(q['gold_parent_id'])
    train, validation, skipped = [], [], []
    for q in queries:
        if q['partition']!='development':
            continue
        number = int(sha256(('ranking-v1:'+q['component_id']).encode()).hexdigest()[:16],16)
        # Whole content components, not individual questions, select checkpoints.
        if number%5==0:
            validation.append(q)
            continue
        if q['query_id'] in forbidden or key(q['question']) in forbidden_text:
            skipped.append({'query_id':q['query_id'],'reason':'evaluation question overlap'})
            continue
        if len(references[key(q['question'])])!=1:
            skipped.append({'query_id':q['query_id'],'reason':'ambiguous duplicate question'})
            continue
        positive = [c for c in by_parent[q['gold_parent_id']] if any(
            c['start']<=s['start'] and s['end']<=c['end'] for s in q['answer_spans'])]
        if not positive:
            skipped.append({'query_id':q['query_id'],'reason':'no fully answer-bearing chunk'})
            continue
        chosen = min(positive,key=lambda c:(len(c['text']),c['start'],c['chunk_id']))
        train.append({'query_id':q['query_id'],'question':q['question'],
                      'parent_id':q['gold_parent_id'],'component_id':q['component_id'],
                      'chunk_id':chosen['chunk_id'],'positive':chosen['text'],'language':q['language']})
    if not train or not validation:
        raise ValueError('Empty ranking train or validation partition')
    if {r['component_id'] for r in train}&{q['component_id'] for q in validation}:
        raise ValueError('Training/validation article leakage')
    if {r['query_id'] for r in train}&forbidden:
        raise ValueError('Evaluation QA leaked into training')
    return train,validation,skipped


def batches(rows, size):
    """Pack unique article components and queries; never make siblings negatives."""
    pending = list(rows)
    while pending:
        batch, rest, parents, components, texts = [], [], set(), set(), set()
        for row in pending:
            q = key(row['question'])
            if (len(batch)>=size or row['parent_id'] in parents
                    or row['component_id'] in components or q in texts):
                rest.append(row)
            else:
                batch.append(row)
                parents.add(row['parent_id']);components.add(row['component_id']);texts.add(q)
        if len(batch)>=2:
            yield batch
        pending = rest


def train_ranked(name, base_path, queries, chunks, evaluate, preserve, work, recipe):
    """Official MNRL + Matryoshka, full-dimensional float32 static export."""
    import gc
    import json
    import math
    import random
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from model2vec import StaticModel
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss,MatryoshkaLoss
    train,validation,skipped = prepare(queries,chunks)
    base = StaticModel.from_pretrained(base_path)
    if base.dim!=1024:
        raise ValueError('Output dimension must remain 1024')
    class ParityEmbedding(StaticEmbedding):
        def __init__(self):
            tokenizer=Tokenizer.from_str(base.tokenizer.to_str());tokenizer.no_truncation()
            super().__init__(tokenizer,embedding_weights=torch.from_numpy(base.embedding.copy()))
        def preprocess(self,inputs,prompt=None,**kwargs):
            if prompt:
                raise ValueError('Static recipe has no instruction prefix')
            ids=[[i for i in x.ids if i!=base.unk_token_id] for x in
                 self.tokenizer.encode_batch(inputs,add_special_tokens=False)]
            if any(not x for x in ids):
                raise ValueError('Empty training tokens')
            return {'input_ids':torch.tensor([i for row in ids for i in row],dtype=torch.long),
                    'offsets':torch.tensor(np.cumsum([0]+[len(row) for row in ids[:-1]]),dtype=torch.long)}
    torch.manual_seed(recipe['seed'])
    module=ParityEmbedding()
    model=SentenceTransformer(modules=[module],device='cuda')
    samples=['Ang Pilipinas ay isang bansa.','The Philippines is an archipelago.','boltahe 123']
    def export():
        return StaticModel(vectors=module.embedding.weight.detach().cpu().numpy().copy(),
            tokenizer=Tokenizer.from_str(module.tokenizer.to_str()),config={**base.config,'normalize':True},normalize=True)
    def parity(static):
        model.eval()
        with torch.no_grad():
            actual=torch.nn.functional.normalize(model(model.tokenize(samples))['sentence_embedding'],dim=1).cpu().numpy()
        expected=static.encode(samples,max_length=None,use_multiprocessing=False)
        delta=float(np.max(np.abs(actual-expected)))
        if delta>2e-6:
            raise ValueError('Train/static export parity mismatch')
        return delta
    before_parity=parity(base)
    criterion=MatryoshkaLoss(model,MultipleNegativesRankingLoss(model),recipe['matryoshka_dimensions'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=0.01)
    generator=random.Random(recipe['seed'])
    epoch_batches=[]
    for epoch in range(recipe['epochs']):
        shuffled=list(train);generator.shuffle(shuffled)
        epoch_batches.append(list(batches(shuffled,recipe['batch_size'])))
    steps=sum(len(b) for b in epoch_batches)
    if not steps:
        raise ValueError('No nonduplicate ranking batches')
    warmup=max(1,math.ceil(steps*recipe['warmup_ratio']))
    def factor(step):
        return min((step+1)/warmup,max(0,(steps-step)/max(1,steps-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    history=[];best=None;best_path=None
    directory=work/(name+'-ranking');directory.mkdir(exist_ok=False)
    def save_json(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
    manifest={'training_query_ids':[r['query_id'] for r in train],
        'validation_query_ids':[q['query_id'] for q in validation],
        'heldout_query_ids':[q['query_id'] for q in queries if q['partition']=='evaluation'],
        'training_pairs':train,'skipped':skipped,'recipe':recipe,
        'negative_policy':'in-batch other unique source components; unjudged negatives may exist',
        'all_question_aggregate_not_heldout':True,'semantic_qa_review':'pending'}
    save_json(directory/'training-manifest.json',manifest)
    preserve.put(directory/'training-manifest.json','reports/'+name+'-ranking-manifest.json')
    for epoch in range(recipe['epochs']+1):
        losses=[]
        if epoch:
            model.train()
            for batch in epoch_batches[epoch-1]:
                optimizer.zero_grad(set_to_none=True)
                features=[{k:v.to('cuda') for k,v in model.tokenize(texts).items()} for texts in
                          ([r['question'] for r in batch],[r['positive'] for r in batch])]
                loss=criterion(features,None)
                if not torch.isfinite(loss):raise ValueError('Nonfinite ranking loss')
                loss.backward();optimizer.step();scheduler.step()
                losses.append(float(loss.detach().cpu()))
        static=export()
        parity_error=parity(static)
        report=evaluate(static,validation)
        value=report['summary']['development_partition']['metrics']['nDCG@10']['mean']
        destination=directory/f'epoch-{epoch:02d}'
        static.save_pretrained(destination)
        preserve.tree(destination,f'models/{name}/epoch-{epoch:02d}')
        save_json(directory/f'validation-{epoch:02d}.json',report)
        preserve.put(directory/f'validation-{epoch:02d}.json',f'reports/{name}-validation-{epoch:02d}.json')
        history.append({'epoch':epoch,'nDCG@10':value,'export_parity':parity_error,
            'training_loss':float(np.mean(losses)) if losses else None,
            'steps':len(losses),'batch_sizes':sorted(set(map(len,epoch_batches[epoch-1]))) if epoch else []})
        if best is None or value>best:
            best,best_path=value,destination
        preserve.event('ranking-epoch-complete',model=name,**history[-1])
        del static,report;gc.collect();torch.cuda.empty_cache()
    final={'model':name,'history':history,'selected_epoch':int(best_path.name.split('-')[1]),
           'selected_by':'development-component validation nDCG@10; baseline epoch0 eligible',
           'train_queries':len(train),'validation_queries':len(validation),'heldout_queries':7922,
           'epochs':recipe['epochs'],'output_dimension':1024,'no_pca':True,
           'before_export_parity':before_parity,'recipe':recipe}
    save_json(directory/'training-report.json',final)
    preserve.put(directory/'training-report.json',f'reports/{name}-ranking-training.json')
    del model,module,base,criterion,optimizer;scheduler=None;gc.collect();torch.cuda.empty_cache()
    return best_path,final
