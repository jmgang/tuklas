"""Independently verify recorded development rankings and immutable AWS output."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

import boto3

from pycon_model2vec_demo.multiwikiqa_evaluation import score, summarize
from pycon_model2vec_demo.multiwikiqa_ranking_v5 import prepare, select_checkpoint


BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'
DEFAULT_JOB = 'pycon-multiwikiqa-teacher-margin-v5-20261004-181701-356b5b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job', default=DEFAULT_JOB)
    parser.add_argument('--arm', choices=('approved_articles', 'external_wikipedia',
                                          'approved_plus_external'), default='approved_articles')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    job = args.job
    arm = args.arm
    prefix = f'pycon-davao-model2vec/{job}/results'
    run = root / 'runs' / job
    plan = json.loads((run / 'mixed-plan.json').read_text())
    session = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1')
    sm, s3 = session.client('sagemaker'), session.client('s3')
    state = sm.describe_training_job(TrainingJobName=job)
    if state['TrainingJobStatus'] != 'Completed':
        raise ValueError('Training job not complete')

    def checked(spec):
        response = s3.get_object(Bucket=BUCKET, Key=spec['key'])
        raw = response['Body'].read()
        response['Body'].close()
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Frozen input hash mismatch: ' + spec['key'])
        return raw

    def saved(relative, index):
        spec = index[relative]
        return checked({'key': prefix + '/' + relative,
                        'bytes': spec['bytes'], 'sha256': spec['sha256']})

    completion_key = ('reports/teacher-margin-probe-completion.json'
                      if arm == 'approved_articles'
                      else 'reports/teacher-margin-arm-completion.json')
    response = s3.get_object(Bucket=BUCKET, Key=prefix + '/' + completion_key)
    raw = response['Body'].read()
    response['Body'].close()
    completion = json.loads(raw)
    if completion['source_plan_sha256'] != sha256((run / 'mixed-plan.json').read_bytes()).hexdigest():
        raise ValueError('Source plan changed')
    index = completion['artifacts']
    head = s3.head_object(Bucket=BUCKET, Key=prefix + '/' + completion_key)
    if head['ContentLength'] != len(raw) or head['Metadata'].get('sha256') != sha256(raw).hexdigest():
        raise ValueError('Completion hash mismatch')
    if completion['heldout_scored'] or completion['heldout_used_for_training_or_selection']:
        raise ValueError('Heldout claim changed')
    layout = json.loads(checked(plan['evaluation_layout']))
    annotations = [json.loads(line) for line in checked(plan['annotations']).splitlines()]
    train, validation, skipped = prepare(annotations, layout['chunks'])
    if (len(train), len(validation), len(skipped)) != (1734, 338, 6):
        raise ValueError('Development cohort changed')
    forbidden = {q['query_id'] for q in annotations if q['partition'] == 'evaluation'}
    manifest = json.loads(saved(f'reports/{arm}_ranked-ranking-manifest.json', index))
    if (len(manifest['training_query_ids']) != 1734
            or len(manifest['validation_query_ids']) != 338
            or set(manifest['heldout_query_ids']) != forbidden
            or set(manifest['training_query_ids']) & forbidden
            or set(manifest['validation_query_ids']) & forbidden
            or len(manifest['mined_triplets']) != 1726):
        raise ValueError('QA training/heldout exclusion changed')
    chunk_positions = {c['chunk_id']: i for i, c in enumerate(layout['chunks'])}
    if len(chunk_positions) != 6546:
        raise ValueError('Chunk IDs changed')
    reports = []
    for epoch in range(4):
        report = json.loads(saved(
            f'reports/{arm}_ranked-validation-{epoch:02d}.json', index))
        if [r['query_id'] for r in report['per_query']] != [q['query_id'] for q in validation]:
            raise ValueError('Recorded development question order changed')
        for q, r in zip(validation, report['per_query']):
            top = r['top10']
            if len(top) != 10 or len({entry['parent_id'] for entry in top}) != 10:
                raise ValueError('Expected ten unique parents')
            ranking = [(entry['parent_id'], chunk_positions[entry['chunk_id']], entry['cosine'])
                       for entry in top]
            if score(q, ranking, layout['chunks']) != r['metrics']:
                raise ValueError('Recorded metric does not reproduce')
        if summarize(report['per_query']) != report['summary']:
            raise ValueError('Aggregate metrics do not reproduce')
        reports.append(report)
    training = json.loads(saved(f'reports/{arm}_ranked-ranking-training.json', index))
    if (training['selected_epoch'] != 3 or select_checkpoint(training['history']) != 3
            or len(training['history']) != 4):
        raise ValueError('Wrong development checkpoint selection')
    baseline = reports[0]['per_query']
    improvement = []
    for epoch, report in enumerate(reports):
        rows = report['per_query']
        by_language = {}
        for language in ('en', 'tl'):
            pairs = [(a, b) for a, b in zip(baseline, rows) if b['language'] == language]
            gained = [b['query_id'] for a, b in pairs if
                      a['metrics']['Recall@10'] == 0 and b['metrics']['Recall@10'] == 1]
            lost = [b['query_id'] for a, b in pairs if
                    a['metrics']['Recall@10'] == 1 and b['metrics']['Recall@10'] == 0]
            n0 = sum(a['metrics']['nDCG@10'] for a, _ in pairs) / len(pairs)
            n1 = sum(b['metrics']['nDCG@10'] for _, b in pairs) / len(pairs)
            if n1 < n0 - 1e-12:
                raise ValueError(f'Epoch {epoch} development nDCG fell in {language}')
            by_language[language] = {'questions': len(pairs),
                                     'baseline_hits': sum(a['metrics']['Recall@10'] for a, _ in pairs),
                                     'candidate_hits': sum(b['metrics']['Recall@10'] for _, b in pairs),
                                     'baseline_nDCG@10': n0, 'candidate_nDCG@10': n1,
                                     'gained_query_ids': gained, 'lost_query_ids': lost}
        improvement.append({'epoch': epoch, 'by_language': by_language})
    weight = f'models/{arm}_ranked/epoch-03/model.safetensors'
    spec = index[weight]
    key = prefix + '/' + weight
    head = s3.head_object(Bucket=BUCKET, Key=key)
    if head['ContentLength'] != spec['bytes'] or head['Metadata'].get('sha256') != spec['sha256']:
        raise ValueError('Selected model S3 byte/hash metadata mismatch')
    def range_bytes(start, end):
        response = s3.get_object(Bucket=BUCKET, Key=key, Range=f'bytes={start}-{end}')
        raw = response['Body'].read()
        response['Body'].close()
        return raw
    length = struct.unpack('<Q', range_bytes(0, 7))[0]
    header = json.loads(range_bytes(8, 7 + length))
    if not any(v.get('shape') == [249999, 1024] and v.get('dtype') == 'F32'
               for name, v in header.items() if name != '__metadata__'):
        raise ValueError('Selected static-vector dimensions/precision changed')
    output = {'job': job, 'arm': arm, 'status': state['TrainingJobStatus'],
              'billable_seconds': state.get('BillableTimeInSeconds'),
              'runtime_compute_estimate_usd': state.get('BillableTimeInSeconds', 0) * 1.315 / 3600,
              'credits_verified': False,
              'source_plan_sha256': completion['source_plan_sha256'],
              'selected_epoch': 3, 'selected_weights_sha256': spec['sha256'],
              'selected_weights_bytes': spec['bytes'],
              'heldout_scored': False, 'heldout_questions_reserved': len(forbidden),
              'development_questions': len(validation),
              'epochs': improvement,
              'verification_scope': 'full hashes of recorded development reports, independent metric recomputation and S3 model checksum header; model body not re-downloaded'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'epochs'}, indent=2))
    print('Epoch 3:', json.dumps(improvement[3]), flush=True)


if __name__ == '__main__':
    main()
