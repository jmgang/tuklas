"""Mine development-train hard negatives from frozen approved-student and teacher vectors.

No evaluation-question vector or label is indexed, ranked, sampled, or emitted.
The existing vectors are read and full-byte checked; no AWS resource is started.
"""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import boto3
import numpy as np

from pycon_model2vec_demo.multiwikiqa_evaluation import normalized
from pycon_model2vec_demo.multiwikiqa_ranking_v3 import prepare


BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'
BASE = 'pycon-davao-model2vec/pycon-multiwikiqa-eval-20261004-131724-611940/results/vectors'
JOB = 'pycon-multiwikiqa-ranked-v3-20261004-165739-f68b9a'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'runs' / JOB / 'mixed-plan.json').read_text())
    s3 = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1').client('s3')

    def checked(spec):
        response = s3.get_object(Bucket=BUCKET, Key=spec['key'])
        raw = response['Body'].read()
        response['Body'].close()
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen byte hash mismatch: ' + spec['key'])
        return raw

    layout = json.loads(checked(plan['evaluation_layout']))
    annotations = [json.loads(x) for x in checked(plan['annotations']).splitlines()]
    train, validation, skipped = prepare(annotations, layout['chunks'])
    if (len(train), len(validation), len(skipped)) != (1734, 338, 6):
        raise ValueError('Frozen split changed')
    if {r['query_id'] for r in train} & {r['query_id'] for r in validation}:
        raise ValueError('Train/validation overlap')
    training_parents = {r['parent_id'] for r in train}
    query_indices = {name: i for i, name in enumerate(layout['query_ids'])}
    if [q['query_id'] for q in annotations] != layout['query_ids']:
        raise ValueError('Query order changed')
    chunk_indices = {chunk['chunk_id']: i for i, chunk in enumerate(layout['chunks'])}
    parent_indices = {name: i for i, name in enumerate(layout['parents'])}
    owners = np.asarray([parent_indices[c['parent_id']] for c in layout['chunks']], dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
    ends = np.r_[starts[1:], len(owners)]
    if len(starts) != 1294 or len(chunk_indices) != 6546:
        raise ValueError('Chunk or parent alignment changed')
    parent_components = {}
    for question in annotations:
        parent = question['gold_parent_id']
        previous = parent_components.setdefault(parent, question['component_id'])
        if previous != question['component_id']:
            raise ValueError('Source article has conflicting component IDs')
    if len(parent_components) != len(layout['parents']):
        raise ValueError('Some source articles lack component identity')

    def vectors(model, role):
        key = f'{BASE}/{model}/{role}.npy'
        response = s3.get_object(Bucket=BUCKET, Key=key)
        raw = response['Body'].read()
        response['Body'].close()
        digest = sha256(raw).hexdigest()
        if response.get('Metadata', {}).get('sha256') != digest:
            raise ValueError('Vector S3 metadata hash mismatch: ' + key)
        array = np.load(BytesIO(raw), allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != (6546 if role == 'chunks' else 10000):
            raise ValueError('Vector shape mismatch: ' + key)
        print(f'Verified {key}, shape={array.shape}', flush=True)
        return normalized(array), digest

    student_docs, sd_hash = vectors('approved_articles', 'chunks')
    teacher_docs, td_hash = vectors('teacher', 'chunks')
    student_queries, sq_hash = vectors('approved_articles', 'queries')
    teacher_queries, tq_hash = vectors('teacher', 'queries')
    if student_docs.shape[1] != teacher_docs.shape[1] or student_docs.shape[1] != 1024:
        raise ValueError('Teacher/student full dimensions changed')
    triplets = []
    omissions = Counter()
    gold_teacher_ranks = []
    for row in train:
        i = query_indices[row['query_id']]
        scores = student_docs @ student_queries[i]
        t_scores = teacher_docs @ teacher_queries[i]
        best = np.asarray([int(a + np.argmax(scores[a:b])) for a, b in zip(starts, ends)])
        order = np.lexsort((np.asarray(layout['parents']), -scores[best]))
        t_best = np.asarray([float(np.max(t_scores[a:b])) for a, b in zip(starts, ends)])
        t_order = np.lexsort((np.asarray(layout['parents']), -t_best))
        gold_parent = parent_indices[row['parent_id']]
        gold_teacher_ranks.append(int(np.flatnonzero(t_order == gold_parent)[0]) + 1)
        positive = chunk_indices[row['chunk_id']]
        positive_t_score = float(t_scores[positive])
        chosen = None
        for p in order[:50]:
            parent = layout['parents'][p]
            if (parent not in training_parents or parent == row['parent_id']
                    or parent_components[parent] == row['component_id']):
                continue
            negative = int(best[p])
            gap = positive_t_score - float(t_scores[negative])
            if gap < .05:
                continue
            chosen = (parent, negative, gap, float(scores[negative]))
            break
        if chosen is None:
            omissions['no_teacher_separated_top50_negative'] += 1
            continue
        parent, negative, gap, student_score = chosen
        triplets.append({'query_id': row['query_id'], 'language': row['language'],
                         'component_id': row['component_id'], 'question': row['question'],
                         'positive_chunk_id': row['chunk_id'], 'positive': row['positive'],
                         'negative_parent_id': parent,
                         'negative_chunk_id': layout['chunks'][negative]['chunk_id'],
                         'negative': layout['chunks'][negative]['text'],
                         'teacher_margin': gap,
                         'teacher_positive_cosine': positive_t_score,
                         'teacher_negative_cosine': float(t_scores[negative]),
                         'student_positive_cosine': float(scores[positive]),
                         'student_negative_cosine': student_score})
    forbidden = {q['query_id'] for q in annotations if q['partition'] == 'evaluation'}
    if {r['query_id'] for r in triplets} & forbidden:
        raise ValueError('Evaluation QA entered training triplets')
    if any(parent_components[r['negative_parent_id']] == r['component_id'] for r in triplets):
        raise ValueError('Same-component false negative entered triplets')
    if any(r['negative_parent_id'] not in training_parents for r in triplets):
        raise ValueError('Validation/heldout source article entered training negatives')
    report = {'scope': 'development TRAIN only; evaluation and validation QA excluded',
              'source_dataset_revision': plan['approved_dataset_revision'] if 'approved_dataset_revision' in plan else
                  plan['source_dataset_revision'] if 'source_dataset_revision' in plan else None,
              'algorithm': 'baseline student top50 unique parents; first non-gold, other-component TRAIN parent chunk with teacher cosine margin >=0.05',
              'negative_source_restriction': 'only development-training source articles; validation and heldout articles excluded',
              'teacher_margin_threshold': .05,
              'train_pairs_available': len(train), 'triplets': len(triplets),
              'omissions': dict(omissions),
              'teacher_gold_parent_top10_on_train': int(sum(rank <= 10 for rank in gold_teacher_ranks)),
              'teacher_gold_parent_top1_on_train': int(sum(rank == 1 for rank in gold_teacher_ranks)),
              'languages': dict(Counter(r['language'] for r in triplets)),
              'unique_negative_parents': len({r['negative_parent_id'] for r in triplets}),
              'input_hashes': {'approved_chunks': sd_hash, 'teacher_chunks': td_hash,
                               'approved_queries': sq_hash, 'teacher_queries': tq_hash,
                               'layout': plan['evaluation_layout']['sha256'],
                               'annotations': plan['annotations']['sha256']},
              'rows': triplets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
