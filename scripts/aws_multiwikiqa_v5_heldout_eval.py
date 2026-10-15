"""Evaluate one frozen, development-selected student on untouched QA.

No training, checkpoint selection, data relabeling or dataset changes occur here.
The historical frozen baseline is compared by query ID on the same layout.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

INPUT = Path('/opt/ml/input/data/input')
WORK = Path('/opt/ml/v5-heldout-evaluation')
BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'


def digest(path):
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    import boto3

    plan = json.loads((INPUT / 'evaluation-plan.json').read_text())
    for filename, spec in plan['source_files'].items():
        path = INPUT / filename
        if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError('Staged source changed: ' + filename)
    sys.path.insert(0, str(INPUT))
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'model2vec==0.9.0', 'numpy==2.5.3'], check=True)
    import numpy as np
    from model2vec import StaticModel
    from multiwikiqa_evaluation import rank_parents, score, summarize, paired_cluster_ci

    np.random.seed(42)
    s3 = boto3.client('s3', region_name='ap-southeast-1')
    WORK.mkdir(exist_ok=False)
    (WORK / 'events').mkdir(exist_ok=False)
    (WORK / 'model').mkdir(exist_ok=False)
    index = {}

    def publish(path, key):
        spec = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        with path.open('rb') as source:
            s3.put_object(Bucket=BUCKET, Key=plan['output_prefix'] + '/' + key,
                          Body=source, ContentLength=spec['bytes'], IfNoneMatch='*',
                          Metadata={'sha256': spec['sha256']})
        index[key] = spec
        return spec

    def save(key, value):
        path = WORK / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as output:
            json.dump(value, output, sort_keys=True, indent=2, ensure_ascii=False)
            output.write('\n')
        publish(path, key)

    def event(stage, **details):
        record = {'at_utc': datetime.now(timezone.utc).isoformat(),
                  'stage': stage, **details}
        save('events/' + uuid4().hex + '.json', record)
        print(json.dumps(record), flush=True)

    def frozen(spec):
        response = s3.get_object(Bucket=BUCKET, Key=spec['key'])
        try:
            raw = response['Body'].read()
        finally:
            response['Body'].close()
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen input checksum mismatch: ' + spec['key'])
        return raw

    layout = json.loads(frozen(plan['layout']))
    questions = [json.loads(line) for line in frozen(plan['annotations']).splitlines()]
    baseline = json.loads(frozen(plan['baseline_report']))
    if (len(questions) != 10000 or len(layout['parents']) != 1294
            or len(layout['chunks']) != 6546
            or [q['query_id'] for q in questions] != layout['query_ids']
            or [r['query_id'] for r in baseline['per_query']] != layout['query_ids']
            or baseline['shared_layout_sha256'] != plan['layout']['sha256']):
        raise ValueError('Frozen evaluation cohort or baseline mismatch')
    if (sum(q['partition'] == 'evaluation' for q in questions) != 7922
            or sum(q['partition'] == 'development' for q in questions) != 2078):
        raise ValueError('Held-out/development partition changed')
    event('frozen-data-verified', questions=10000, heldout=7922,
          parents=1294, chunks=6546)

    for filename, spec in plan['selected_model'].items():
        target = WORK / 'model' / filename
        response = s3.get_object(Bucket=BUCKET, Key=spec['key'])
        value = sha256()
        try:
            with target.open('xb') as output:
                for block in iter(lambda: response['Body'].read(2**20), b''):
                    value.update(block)
                    output.write(block)
        finally:
            response['Body'].close()
        if target.stat().st_size != spec['bytes'] or value.hexdigest() != spec['sha256']:
            raise ValueError('Selected model bytes changed: ' + filename)
    model = StaticModel.from_pretrained(str(WORK / 'model'))
    if model.embedding.shape != (249999, 1024) or model.embedding.dtype != np.float32:
        raise ValueError('Frozen student capacity or precision changed')
    event('selected-model-verified', selected_epoch=plan['selected_epoch'],
          weights_sha256=plan['selected_model']['model.safetensors']['sha256'])

    chunks, ids = layout['chunks'], layout['parents']
    position = {item: i for i, item in enumerate(ids)}
    owners = [position[chunk['parent_id']] for chunk in chunks]
    documents = np.asarray(model.encode([c['text'] for c in chunks],
                                        batch_size=512, max_length=None), dtype=np.float32)
    queries = np.asarray(model.encode([q['question'] for q in questions],
                                      batch_size=512, max_length=None), dtype=np.float32)
    if (documents.shape != (6546, 1024) or queries.shape != (10000, 1024)
            or not np.isfinite(documents).all() or not np.isfinite(queries).all()):
        raise ValueError('Encoded vectors invalid')
    event('encoding-complete', passage_vectors=6546, query_vectors=10000)

    results = []
    for start in range(0, len(questions), 64):
        rankings = rank_parents(queries[start:start + 64], documents, owners, ids)
        for question, ranking in zip(questions[start:start + 64], rankings):
            results.append({k: question[k] for k in
                            ('query_id', 'gold_parent_id', 'language', 'partition', 'exclusion_reasons')}
                           | {'metrics': score(question, ranking, chunks),
                              'top10': [{'parent_id': parent,
                                         'chunk_id': chunks[index]['chunk_id'],
                                         'cosine': cosine}
                                        for parent, index, cosine in ranking]})
    event('scoring-complete', questions=len(results))

    def cohort(rows, language):
        return [r for r in rows if r['partition'] == 'evaluation'
                and (language == 'all' or r['language'] == language)]

    comparisons = {}
    for language in ('all', 'en', 'tl'):
        before = cohort(baseline['per_query'], language)
        after = cohort(results, language)
        if [r['query_id'] for r in before] != [r['query_id'] for r in after]:
            raise ValueError('Frozen paired cohort mismatch')
        gained = [a['query_id'] for a, b in zip(after, before)
                  if a['metrics']['Recall@10'] > b['metrics']['Recall@10']]
        lost = [a['query_id'] for a, b in zip(after, before)
                if a['metrics']['Recall@10'] < b['metrics']['Recall@10']]
        comparisons[language] = {
            'questions': len(after),
            'before': summarize(before)['evaluation_partition']['metrics'],
            'after': summarize(after)['evaluation_partition']['metrics'],
            'gained_recall10_query_ids': gained,
            'lost_recall10_query_ids': lost,
            'paired_article_ci': {key: paired_cluster_ci(after, before, key)
                                  for key in ('Recall@10', 'MRR@10', 'nDCG@10')},
        }
    promotion = all(
        not comparisons[language]['lost_recall10_query_ids']
        and all(comparisons[language]['after'][key]['mean'] >=
                comparisons[language]['before'][key]['mean']
                for key in ('Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
                            'MRR@10', 'nDCG@10'))
        for language in ('en', 'tl'))
    report = {
        'model': plan['candidate_label'],
        'baseline': plan['baseline_label'],
        'status': 'evaluation-complete',
        'promotion_gate_passed': promotion,
        'promotion_gate': 'No lost Recall@10 question and no decline in any retrieval metric in either language',
        'heldout_questions': 7922, 'full_questions': 10000,
        'layout_sha256': plan['layout']['sha256'],
        'annotations_sha256': plan['annotations']['sha256'],
        'selected_weights_sha256': plan['selected_model']['model.safetensors']['sha256'],
        'selected_epoch': plan['selected_epoch'],
        'training_and_selection_excluded_heldout': True,
        'per_query': results,
        'summary': summarize(results),
        'heldout_comparisons': comparisons,
        'model_weight_bytes': int(model.embedding.nbytes),
        'passage_vector_bytes': int(documents.nbytes),
        'query_vector_bytes': int(queries.nbytes),
        'latency_measured': False,
        'runtime_memory_measured': False,
        'limitations': ['As-released synthetic QA; semantic review pending',
                        'One known source article per question, not exhaustive relevance',
                        'No answer generation evaluated'],
    }
    save('reports/heldout-quality.json', report)
    save('reports/completion.json', {
        'status': 'complete', 'promotion_gate_passed': promotion,
        'heldout_questions': 7922, 'artifacts': dict(index),
        'source_plan_sha256': sha256((INPUT / 'evaluation-plan.json').read_bytes()).hexdigest()})
    event('complete', promotion_gate_passed=promotion)


if __name__ == '__main__':
    main()
