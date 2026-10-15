"""One additional frozen-student arm with the verified teacher-margin recipe.

The shared audited negatives were mined on the approved student, but every
negative parent is in the development-training partition. No heldout QA is
encoded, ranked, used for optimization or checkpoint selection.
"""

from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys

INPUT = Path('/opt/ml/input/data/input')
sys.path.insert(0, str(INPUT))
os.environ['SEALION_INPUT_DIR'] = str(INPUT)
os.environ['SEALION_WORK_DIR'] = '/opt/ml/teacher-margin-arm-work'
original = importlib.import_module('aws_sealion_train_repair_v3')
CONFIG = json.loads((INPUT / 'settings.json').read_text())
PLAN = json.loads((INPUT / 'mixed-plan.json').read_text())


def main():
    arm = PLAN['ranking_arm']
    if arm not in ('external_wikipedia', 'approved_plus_external'):
        raise ValueError('Only two remaining frozen arms may be run')
    original.WORK.mkdir(parents=True, exist_ok=False)
    (original.WORK / 'events').mkdir(exist_ok=False)
    preserve = original.Preservation()
    for filename, spec in PLAN['source_files'].items():
        path = INPUT / filename
        if path.stat().st_size != spec['bytes'] or original.digest(path) != spec['sha256']:
            raise ValueError('Staged source changed: ' + filename)
    preserve.event('dependency-setup', arm=arm)
    original.bootstrap()
    from sealion_runtime import verify_runtime
    verify_runtime(CONFIG['tokenlearn']['commit'])
    import numpy as np
    import torch
    from multiwikiqa_evaluation import rank_parents, score, summarize
    from multiwikiqa_ranking import train_ranked, prepare

    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for frozen ranking protocol')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(8)
    bucket = PLAN['bucket']

    def get(spec):
        response = preserve.s3.get_object(Bucket=bucket, Key=spec['key'])
        try:
            raw = response['Body'].read()
        finally:
            response['Body'].close()
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen source checksum mismatch: ' + spec['key'])
        return raw

    layout = json.loads(get(PLAN['evaluation_layout']))
    questions = [json.loads(line) for line in get(PLAN['annotations']).splitlines()]
    if (len(questions) != 10000 or len(layout['chunks']) != 6546
            or len(layout['parents']) != 1294
            or [q['query_id'] for q in questions] != layout['query_ids']):
        raise ValueError('Frozen question or corpus cohort changed')
    train, validation, skipped = prepare(questions, layout['chunks'])
    if (len(train), len(validation), len(skipped)) != (1734, 338, 6):
        raise ValueError('Frozen development split changed')
    chunks, parent_ids = layout['chunks'], layout['parents']
    positions = {parent: i for i, parent in enumerate(parent_ids)}
    owners = [positions[chunk['parent_id']] for chunk in chunks]

    def evaluate(model, cohort):
        if any(q['partition'] != 'development' for q in cohort):
            raise ValueError('Heldout QA may not be scored during selection')
        documents = model.encode([c['text'] for c in chunks], batch_size=512, max_length=None)
        query_vectors = model.encode([q['question'] for q in cohort],
                                     batch_size=512, max_length=None)
        rows = []
        for start in range(0, len(cohort), 64):
            ranks = rank_parents(query_vectors[start:start + 64],
                                 documents, owners, parent_ids)
            for question, ranking in zip(cohort[start:start + 64], ranks):
                rows.append({k: question[k] for k in
                             ('query_id', 'gold_parent_id', 'language', 'partition', 'exclusion_reasons')}
                            | {'metrics': score(question, ranking, chunks),
                               'top10': [{'parent_id': parent,
                                          'chunk_id': chunks[index]['chunk_id'],
                                          'cosine': cosine}
                                         for parent, index, cosine in ranking]})
        return {'question_count': len(rows), 'per_query': rows,
                'summary': summarize(rows),
                'vector_bytes': documents.nbytes,
                'query_vector_bytes': query_vectors.nbytes}

    mined = json.loads((INPUT / 'mined-triplets.json').read_text())
    if (mined['triplets'] != 1726 or mined['train_pairs_available'] != 1734
            or mined['teacher_margin_threshold'] != .05
            or mined['negative_source_restriction'] !=
            'only development-training source articles; validation and heldout articles excluded'):
        raise ValueError('Audited development-only negative set changed')
    specs = (PLAN['frozen_students'][arm] if arm == 'external_wikipedia'
             else PLAN['recovered_mixture'])
    model_path = original.WORK / ('frozen-' + arm)
    model_path.mkdir(exist_ok=False)
    for filename, spec in specs.items():
        response = preserve.s3.get_object(Bucket=bucket, Key=spec['key'])
        h = sha256()
        try:
            with (model_path / filename).open('xb') as output:
                for block in iter(lambda: response['Body'].read(2**20), b''):
                    h.update(block)
                    output.write(block)
        finally:
            response['Body'].close()
        if (model_path / filename).stat().st_size != spec['bytes'] or h.hexdigest() != spec['sha256']:
            raise ValueError('Frozen student checksum mismatch: ' + filename)
    preserve.event('frozen-student-restored', arm=arm, all_bytes_hashed=True)
    _, report = train_ranked(arm + '_ranked', model_path, questions, chunks,
                             evaluate, preserve, original.WORK,
                             PLAN['ranking'], mined)
    summary = {'status': 'development-probe-complete', 'arm': arm,
               'objective': 'full-1024 teacher-margin loss, audited shared train-only negatives',
               'negative_mining_base': 'approved_articles',
               'training_report': report,
               'heldout_scored': False,
               'heldout_used_for_training_or_selection': False,
               'artifacts': preserve.index,
               'source_plan_sha256': original.digest(INPUT / 'mixed-plan.json')}
    destination = original.WORK / 'teacher-margin-arm-completion.json'
    original.write_json(destination, summary)
    preserve.put(destination, 'reports/teacher-margin-arm-completion.json')
    preserve.event('teacher-margin-arm-complete', arm=arm,
                   selected_epoch=report['selected_epoch'], heldout_scored=False)


if __name__ == '__main__':
    main()
