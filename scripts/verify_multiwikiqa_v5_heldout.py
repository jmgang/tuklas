"""Independently re-score an immutable held-out report; no model rerun."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import boto3

from pycon_model2vec_demo.multiwikiqa_evaluation import score, summarize

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    job = args.job
    prefix = 'pycon-davao-model2vec/' + job + '/results'
    aws = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1')
    sm, s3 = aws.client('sagemaker'), aws.client('s3')
    state = sm.describe_training_job(TrainingJobName=job)
    if state['TrainingJobStatus'] != 'Completed':
        raise ValueError('Evaluation job not complete')
    plan = json.loads((ROOT / 'runs' / job / 'evaluation-plan.json').read_text())

    def get(spec):
        response = s3.get_object(Bucket=BUCKET, Key=spec['key'])
        try:
            raw = response['Body'].read()
        finally:
            response['Body'].close()
        if len(raw) != spec['bytes'] or sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Source checksum mismatch: ' + spec['key'])
        return raw

    def result(key):
        response = s3.get_object(Bucket=BUCKET, Key=prefix + '/' + key)
        try:
            raw = response['Body'].read()
        finally:
            response['Body'].close()
        head = s3.head_object(Bucket=BUCKET, Key=prefix + '/' + key)
        if (len(raw) != head['ContentLength']
                or sha256(raw).hexdigest() != head['Metadata'].get('sha256')):
            raise ValueError('Result checksum mismatch: ' + key)
        return json.loads(raw), {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}

    layout = json.loads(get(plan['layout']))
    queries = [json.loads(line) for line in get(plan['annotations']).splitlines()]
    baseline = json.loads(get(plan['baseline_report']))
    report, report_spec = result('reports/heldout-quality.json')
    completion, _ = result('reports/completion.json')
    if (completion['status'] != 'complete'
            or completion['artifacts']['reports/heldout-quality.json'] != report_spec
            or completion['source_plan_sha256'] != sha256(
                (ROOT / 'runs' / job / 'evaluation-plan.json').read_bytes()).hexdigest()):
        raise ValueError('Completion index or plan hash mismatch')
    if (len(queries), len(report['per_query']), len(layout['chunks']),
            len(layout['parents'])) != (10000, 10000, 6546, 1294):
        raise ValueError('Frozen dimensions changed')
    if ([q['query_id'] for q in queries] != layout['query_ids']
            or [r['query_id'] for r in report['per_query']] != layout['query_ids']
            or [r['query_id'] for r in baseline['per_query']] != layout['query_ids']):
        raise ValueError('Question order or identity changed')
    if report['selected_weights_sha256'] != plan['selected_model']['model.safetensors']['sha256']:
        raise ValueError('Selected model identity changed')
    by_id = {c['chunk_id']: (i, c) for i, c in enumerate(layout['chunks'])}
    for q, row, previous in zip(queries, report['per_query'], baseline['per_query']):
        if ((q['query_id'], q['gold_parent_id'], q['language'], q['partition']) !=
                (row['query_id'], row['gold_parent_id'], row['language'], row['partition'])):
            raise ValueError('Question metadata changed')
        if (q['query_id'], q['gold_parent_id']) != (previous['query_id'], previous['gold_parent_id']):
            raise ValueError('Baseline cohort changed')
        ranking = []
        for item in row['top10']:
            index, chunk = by_id[item['chunk_id']]
            if chunk['parent_id'] != item['parent_id']:
                raise ValueError('Returned chunk belongs to different parent')
            ranking.append((item['parent_id'], index, item['cosine']))
        if len(ranking) != 10 or len({r[0] for r in ranking}) != 10:
            raise ValueError('Top ten did not contain ten unique source articles')
        computed = score(q, ranking, layout['chunks'])
        if computed != row['metrics']:
            raise ValueError('Per-query metrics inconsistent: ' + q['query_id'])
    derived = summarize(report['per_query'])
    if derived != report['summary']:
        raise ValueError('Saved aggregate metrics inconsistent')
    comparison = {}
    for language in ('all', 'en', 'tl'):
        rows = [r for r in report['per_query'] if r['partition'] == 'evaluation'
                and (language == 'all' or r['language'] == language)]
        before = [r for r in baseline['per_query'] if r['partition'] == 'evaluation'
                  and (language == 'all' or r['language'] == language)]
        if [r['query_id'] for r in rows] != [r['query_id'] for r in before]:
            raise ValueError('Heldout pair mismatch')
        saved = report['heldout_comparisons'][language]
        got_before = summarize(before)['evaluation_partition']['metrics']
        got_after = summarize(rows)['evaluation_partition']['metrics']
        if saved['before'] != got_before or saved['after'] != got_after:
            raise ValueError('Paired summary mismatch')
        gains = [r['query_id'] for r, b in zip(rows, before)
                 if r['metrics']['Recall@10'] > b['metrics']['Recall@10']]
        losses = [r['query_id'] for r, b in zip(rows, before)
                  if r['metrics']['Recall@10'] < b['metrics']['Recall@10']]
        if (saved['gained_recall10_query_ids'] != gains
                or saved['lost_recall10_query_ids'] != losses):
            raise ValueError('Per-query gains/losses mismatch')
        comparison[language] = {'questions': len(rows),
                                'baseline_recall10': got_before['Recall@10']['mean'],
                                'candidate_recall10': got_after['Recall@10']['mean'],
                                'baseline_ndcg10': got_before['nDCG@10']['mean'],
                                'candidate_ndcg10': got_after['nDCG@10']['mean'],
                                'gained': len(gains), 'lost': len(losses)}
    gate = all(
        not report['heldout_comparisons'][language]['lost_recall10_query_ids']
        and all(report['heldout_comparisons'][language]['after'][metric]['mean'] >=
                report['heldout_comparisons'][language]['before'][metric]['mean']
                for metric in ('Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
                               'MRR@10', 'nDCG@10'))
        for language in ('en', 'tl'))
    if report['promotion_gate_passed'] != gate or completion['promotion_gate_passed'] != gate:
        raise ValueError('Strict no-regression gate inconsistent')
    out = {
        'status': 'independently-verified',
        'job': job, 'training_job': plan['source_job'],
        'sagemaker_status': state['TrainingJobStatus'],
        'billable_seconds': state['BillableTimeInSeconds'],
        'compute_estimate_usd': state['BillableTimeInSeconds'] * 1.315 / 3600,
        'credits_verified': False,
        'report': report_spec,
        'model_weights_sha256': report['selected_weights_sha256'],
        'full_questions_recomputed': 10000,
        'heldout_questions': 7922,
        'top_ten_unique_parent_ids_verified': True,
        'heldout_results': comparison,
        'strict_zero_loss_gate_passed': gate,
        'aggregate_nonregression_all_reported_metrics': all(
            report['heldout_comparisons'][language]['after'][metric]['mean'] >=
            report['heldout_comparisons'][language]['before'][metric]['mean']
            for language in ('en', 'tl')
            for metric in ('Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
                           'MRR@10', 'nDCG@10')),
        'scope': 'Verifies S3 report bytes, recorded rankings, original annotations and metrics; does not independently re-encode model',
    }
    target = ROOT / 'runs' / job / 'independent-heldout-verification-v1.json'
    with target.open('x') as output:
        json.dump(out, output, sort_keys=True, indent=2)
        output.write('\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
