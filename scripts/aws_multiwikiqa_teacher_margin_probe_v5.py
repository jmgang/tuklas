"""One-arm, development-only teacher-guided hard-negative ranking probe.

The approved student, split, learning rate, epochs and evaluator stay frozen.
The only new training signal is SEA-LION's positive-minus-negative cosine
margin on audited development-training triplets. No held-out scoring.
"""

import gc
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys

INPUT = Path('/opt/ml/input/data/input')
sys.path.insert(0, str(INPUT))
os.environ['SEALION_INPUT_DIR'] = str(INPUT)
os.environ['SEALION_WORK_DIR'] = '/opt/ml/teacher-margin-probe-work'
original = importlib.import_module('aws_sealion_train_repair_v3')
CONFIG = json.loads((INPUT / 'settings.json').read_text())
PLAN = json.loads((INPUT / 'mixed-plan.json').read_text())


def main():
    original.WORK.mkdir(parents=True, exist_ok=False)
    (original.WORK / 'events').mkdir(exist_ok=False)
    preserve = original.Preservation()
    for name, spec in PLAN['source_files'].items():
        path = INPUT / name
        if path.stat().st_size != spec['bytes'] or original.digest(path) != spec['sha256']:
            raise ValueError('Staged source changed: ' + name)
    preserve.event('dependency-setup')
    original.bootstrap()
    from sealion_runtime import verify_runtime
    verify_runtime(CONFIG['tokenlearn']['commit'])
    import numpy as np
    import torch
    from model2vec import StaticModel
    from multiwikiqa_evaluation import rank_parents, score, summarize
    from multiwikiqa_ranking import train_ranked, prepare
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this objective probe')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    bucket = PLAN['bucket']

    def get(spec):
        response = preserve.s3.get_object(Bucket=bucket, Key=spec['key'])
        try:
            raw = response['Body'].read()
        finally:
            response['Body'].close()
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen input byte hash changed: ' + spec['key'])
        return raw

    layout = json.loads(get(PLAN['evaluation_layout']))
    queries = [json.loads(line) for line in get(PLAN['annotations']).splitlines()]
    if (len(queries) != 10000 or len(layout['chunks']) != 6546
            or len(layout['parents']) != 1294
            or [q['query_id'] for q in queries] != layout['query_ids']):
        raise ValueError('Frozen cohort or layout changed')
    chunks, ids = layout['chunks'], layout['parents']
    train, validation, skipped = prepare(queries, chunks)
    if (len(train), len(validation), len(skipped)) != (1734, 338, 6):
        raise ValueError('Development split changed')
    if any(q['partition'] == 'evaluation' for q in validation):
        raise ValueError('Heldout query entered development evaluation')
    positions = {name: i for i, name in enumerate(ids)}
    owners = [positions[c['parent_id']] for c in chunks]

    def evaluate(model, cohort):
        if any(q['partition'] != 'development' for q in cohort):
            raise ValueError('Only development questions may be evaluated in this job')
        documents = model.encode([c['text'] for c in chunks], batch_size=512, max_length=None)
        questions = model.encode([q['question'] for q in cohort], batch_size=512, max_length=None)
        rows = []
        for start in range(0, len(cohort), 64):
            ranks = rank_parents(questions[start:start + 64], documents, owners, ids)
            for question, ranking in zip(cohort[start:start + 64], ranks):
                rows.append({k: question[k] for k in
                             ('query_id', 'gold_parent_id', 'language', 'partition', 'exclusion_reasons')}
                            | {'metrics': score(question, ranking, chunks),
                               'top10': [{'parent_id': parent, 'chunk_id': chunks[index]['chunk_id'],
                                          'cosine': cosine} for parent, index, cosine in ranking]})
        return {'question_count': len(cohort), 'per_query': rows, 'summary': summarize(rows),
                'vector_bytes': documents.nbytes, 'query_vector_bytes': questions.nbytes}

    mined = json.loads((INPUT / 'mined-triplets.json').read_text())
    if (mined['triplets'] != 1726 or mined['train_pairs_available'] != 1734
            or mined['teacher_margin_threshold'] != .05
            or mined['negative_source_restriction'] !=
            'only development-training source articles; validation and heldout articles excluded'):
        raise ValueError('Frozen hard-negative mining audit changed')
    preserve.event('mined-triplets-audited', triplets=1726, teacher_margin_threshold=.05,
                   development_training_articles_only=True)
    arm = 'approved_articles'
    path = original.WORK / 'frozen-approved-articles'
    path.mkdir(exist_ok=False)
    for filename, spec in PLAN['frozen_students'][arm].items():
        response = preserve.s3.get_object(Bucket=bucket, Key=spec['key'])
        h = sha256()
        with (path / filename).open('xb') as output:
            for block in iter(lambda: response['Body'].read(2**20), b''):
                h.update(block)
                output.write(block)
        response['Body'].close()
        if (path / filename).stat().st_size != spec['bytes'] or h.hexdigest() != spec['sha256']:
            raise ValueError('Frozen student changed: ' + filename)
    preserve.event('frozen-student-restored', arm=arm, all_bytes_hashed=True)
    selected, report = train_ranked('approved_articles_ranked', path, queries, chunks,
                                    evaluate, preserve, original.WORK, PLAN['ranking'], mined)
    summary = {'status': 'development-probe-complete',
               'objective': 'full-1024 MarginMSELoss with SEA-LION margin and audited hard negatives',
               'change_from_v4': 'teacher-guided train-article triplets replace positive-only MNRL; frozen student/split/LR/epochs unchanged',
               'training_report': report,
               'selected_checkpoint_s3': (preserve.prefix + '/models/approved_articles_ranked/'
                                          + selected.name),
               'heldout_scored': False,
               'heldout_used_for_training_or_selection': False,
               'artifacts': preserve.index,
               'source_plan_sha256': original.digest(INPUT / 'mixed-plan.json')}
    target = original.WORK / 'teacher-margin-probe-completion.json'
    original.write_json(target, summary)
    preserve.put(target, 'reports/teacher-margin-probe-completion.json')
    preserve.event('teacher-margin-probe-complete', selected_epoch=report['selected_epoch'],
                   heldout_scored=False)
    del report, selected
    gc.collect()


if __name__ == '__main__':
    main()
