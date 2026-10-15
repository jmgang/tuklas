"""Read-only frozen-result diagnosis. No model updates or new compute."""
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pycon_model2vec_demo.multiwikiqa_evaluation import paired_cluster_ci


def main():
    s3 = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1').client('s3')
    bucket = 'sagemaker-ap-southeast-1-<aws-account-id>'
    prefix = 'pycon-davao-model2vec/pycon-multiwikiqa-eval-20261004-131724-611940/results/'
    complete = json.loads(s3.get_object(Bucket=bucket, Key=prefix+'reports/completion.json')['Body'].read())
    def fetch(name):
        raw = s3.get_object(Bucket=bucket, Key=prefix+name)['Body'].read()
        spec = complete['artifacts'][name]
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen artifact mismatch: '+name)
        return json.loads(raw)
    reports = {n: fetch('reports/'+n+'.json') for n in complete['models']}
    annotations = [json.loads(x) for x in (ROOT/'runs/multiwikiqa-preparation-20261004-v1/annotations.all.jsonl').read_text().splitlines()]
    parents = {r['parent_id']: r for r in map(json.loads, (ROOT/'runs/multiwikiqa-preparation-20261004-v1/parents.jsonl').read_text().splitlines())}
    for r in reports.values():
        if [x['query_id'] for x in r['per_query']] != [q['query_id'] for q in annotations]:
            raise ValueError('Cohort changed')
    audit = {'frozen_reports_sha256_checked': True, 'no_retraining': True,
        'initialization_retrieval_baseline_evaluated': False, 'corpus_mixture_evaluated': False,
        'language_overlap_counts': {}, 'language_paired_intervals': {}, 'tagalog_examples': {}}
    names = ('approved_articles', 'external_wikipedia', 'potion_retrieval')
    for language, code in (('English','en'), ('Tagalog','tl')):
        filtered = {n: [r for r in reports[n]['per_query'] if r['language']==code] for n in names}
        audit['language_paired_intervals'][language] = {
            a+' minus '+b: {metric: paired_cluster_ci(filtered[a], filtered[b], metric)
                for metric in ('Recall@10', 'nDCG@10')}
            for a,b in [('approved_articles','external_wikipedia'), ('approved_articles','potion_retrieval')]}
        audit['language_overlap_counts'][language] = {}
        for a,b in [('approved_articles','external_wikipedia'), ('approved_articles','potion_retrieval')]:
            counts = Counter()
            for x,y in zip(filtered[a],filtered[b]):
                yesx = bool(x['metrics']['Recall@10'])
                yesy = bool(y['metrics']['Recall@10'])
                counts['both_found' if yesx and yesy else 'left_only' if yesx else 'right_only' if yesy else 'neither_found'] += 1
            audit['language_overlap_counts'][language][a+' versus '+b] = dict(counts)
    for label, good, bad in [('potion_found_approved_missed','potion_retrieval','approved_articles'),
                              ('approved_found_potion_missed','approved_articles','potion_retrieval')]:
        examples = []
        for q,a,b in zip(annotations, reports[good]['per_query'],reports[bad]['per_query']):
            if q['language']!='tl' or not a['metrics']['Recall@10'] or b['metrics']['Recall@10']:
                continue
            examples.append({'query_id':q['query_id'],'question':q['question'],
                'gold_parent_id':q['gold_parent_id'], 'gold_title':parents[q['gold_parent_id']].get('title'),
                good+'_rank':next(i+1 for i,r in enumerate(a['top10']) if r['parent_id']==q['gold_parent_id']),
                bad+'_top1_title':parents[b['top10'][0]['parent_id']].get('title')})
            if len(examples)==4:
                break
        audit['tagalog_examples'][label] = examples
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT/'models/potion-retrieval-6fc8051fab2a/tokenizer.json'))
    tokenizer.no_truncation()
    unknown_id = tokenizer.token_to_id('[UNK]')
    stats = {}
    for language, code in (('English','en'), ('Tagalog','tl')):
        questions = [q['question'] for q in annotations if q['language']==code]
        encodings = tokenizer.encode_batch(questions, add_special_tokens=False)
        total = sum(len(x.ids) for x in encodings)
        unknown = sum(x.ids.count(unknown_id) for x in encodings) if unknown_id is not None else None
        stats[language] = {'questions':len(questions),'total_tokens':total,'unknown_tokens':unknown,
            'mean_tokens_per_question':total/len(questions),
            'questions_with_unknown':sum(unknown_id in x.ids for x in encodings) if unknown_id is not None else None}
    audit['potion_query_tokenization'] = {'vocabulary_size':tokenizer.get_vocab_size(),
        'unknown_id':unknown_id, 'by_language':stats,
        'interpretation':'Token coverage is not proof of semantic language understanding.'}
    audit['training'] = {}
    for name in ('approved_articles','external_wikipedia'):
        p = ROOT/'runs/pycon-sealion-repair-20261004-084939-91585a/verification-20261004T113545Z'/f'{name}-training-report.json'
        r = json.loads(p.read_text())
        history = [v['val_loss'] for v in r['validation_history']]
        audit['training'][name] = {k:r[k] for k in ('training_rows','validation_rows','training_seconds','dimension','no_pca')}
        audit['training'][name].update({'completed_epochs':len(history), 'first_validation_loss':history[0],
            'best_validation_loss':min(history),'final_validation_loss':history[-1],
            'batch_size':r['training_recipe']['batch_size'],
            'optimizer_steps_at_full_epoch_cadence':len(history)*((r['training_rows']+255)//256)})
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
