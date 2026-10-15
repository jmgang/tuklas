"""Launch one bounded, approved-arm, development-only loss-objective probe."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import boto3

ROOT = Path(__file__).resolve().parents[1]
PRIOR = 'pycon-multiwikiqa-ranked-v3-20261004-165739-f68b9a'
BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'


def digest(path):
    h = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2, default=str)
        output.write('\n')


def main():
    session = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1')
    if session.client('sts').get_caller_identity()['Account'] != '<aws-account-id>':
        raise ValueError('Wrong AWS account')
    sm, s3 = session.client('sagemaker'), session.client('s3')
    state = sm.describe_training_job(TrainingJobName=PRIOR)
    if state['TrainingJobStatus'] != 'Stopped':
        raise ValueError('V3 must remain stopped')
    for match in ('pycon-multiwikiqa', 'pycon-sealion'):
        for page in sm.get_paginator('list_training_jobs').paginate(
                NameContains=match, StatusEquals='InProgress'):
            if page['TrainingJobSummaries']:
                raise ValueError('Another experimental training job is active')

    old = ROOT / 'runs' / PRIOR
    plan = json.loads((old / 'mixed-plan.json').read_text())
    if (plan['ranking']['learning_rate'] != .0005 or
            plan['ranking']['matryoshka_dimensions'] != [32, 64, 128, 256, 512, 1024] or
            plan['ranking_preflight']['validation_questions'] != 338):
        raise ValueError('Controlled predecessor protocol changed')
    name = ('pycon-multiwikiqa-objective-v4-' +
            datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:6])
    directory = ROOT / 'runs' / name
    directory.mkdir(exist_ok=False)
    prefix = 'pycon-davao-model2vec/' + name
    archive = ROOT / 'runs/aws-demo-source-index-20261004T132400Z/exact/code/pycon-sealion-repair-20261004-084939-91585a'
    files = {
        'aws_multiwikiqa_objective_probe_v4.py': ROOT / 'scripts/aws_multiwikiqa_objective_probe_v4.py',
        'aws_sealion_train_repair_v3.py': archive / 'repair/aws_sealion_train_repair_v3.py',
        'sealion_runtime.py': archive / 'repair/sealion_runtime.py',
        'sealion_source_integrity.py': archive / 'repair/sealion_source_integrity.py',
        'sealion_training.py': archive / 'input/sealion_training.py',
        'multiwikiqa_evaluation.py': ROOT / 'src/pycon_model2vec_demo/multiwikiqa_evaluation.py',
        'multiwikiqa_ranking.py': ROOT / 'src/pycon_model2vec_demo/multiwikiqa_ranking_v4.py',
    }
    for path in files.values():
        compile(path.read_text(), str(path), 'exec')
    save(directory / 'settings.json', json.loads((old / 'settings.json').read_text()))
    files['settings.json'] = directory / 'settings.json'
    plan['repair'] = ('Development-only controlled objective test after v3 Tagalog Recall loss; '
                      'same approved frozen student, pairs, validation split, LR, epochs, evaluator. '
                      'No heldout quality or checkpoint selection.')
    plan['ranking'] = {**plan['ranking'], 'matryoshka_dimensions': [1024],
                       'method': 'Full-1024 MNRL objective; no smaller-dimension auxiliary losses',
                       'selection': 'Strict non-regression in both languages; stop immediately if development metric falls'}
    plan['source_files'] = {filename: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                            for filename, path in files.items()}
    save(directory / 'mixed-plan.json', plan)
    files['mixed-plan.json'] = directory / 'mixed-plan.json'
    for filename, path in files.items():
        with path.open('rb') as source:
            s3.put_object(Bucket=BUCKET, Key=prefix + '/input/' + filename, Body=source,
                          IfNoneMatch='*', Metadata={'sha256': digest(path)})
        head = s3.head_object(Bucket=BUCKET, Key=prefix + '/input/' + filename)
        if head['ContentLength'] != path.stat().st_size or head['Metadata'].get('sha256') != digest(path):
            raise ValueError('Staged source checksum mismatch')
    request = json.loads((old / 'create-training-job-request.json').read_text())
    request['TrainingJobName'] = name
    request['AlgorithmSpecification']['ContainerArguments'] = [
        'python /opt/ml/input/data/input/aws_multiwikiqa_objective_probe_v4.py']
    request['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri'] = 's3://' + BUCKET + '/' + prefix + '/input/'
    request['OutputDataConfig']['S3OutputPath'] = 's3://' + BUCKET + '/' + prefix + '/native-output/'
    request['StoppingCondition']['MaxRuntimeInSeconds'] = 1800
    request['Environment']['SEALION_OUTPUT_PREFIX'] = prefix + '/results'
    request['Tags'] = [{'Key': 'Project', 'Value': 'pycon-davao-model2vec-demo'},
                       {'Key': 'Purpose', 'Value': 'development-only-objective-probe'},
                       {'Key': 'DemoSourceManifest', 'Value': 's3://' + BUCKET + '/' + prefix + '/input/mixed-plan.json'}]
    save(directory / 'create-training-job-request.json', request)
    save(directory / 'resource-record.json', {
        'job_name': name, 'region': 'ap-southeast-1', 'profile': 'admin-general',
        's3_results': 's3://' + BUCKET + '/' + prefix + '/results/',
        'instance': 'ml.g4dn.2xlarge', 'max_runtime_seconds': 1800,
        'rate_usd_per_hour': 1.315, 'max_runtime_compute_estimate_usd': .6575,
        'credits_verified': False, 'no_endpoint': True, 'managed_release_on_terminal': True,
        'frozen_predecessor': PRIOR, 'heldout_scored': False,
        'only_change': 'Matryoshka dimensions [32,64,128,256,512,1024] to [1024]',
        'source_files': {filename: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                         for filename, path in files.items()}})
    response = sm.create_training_job(**request)
    save(directory / 'launch-response.json', response)
    save(directory / 'initial-job-state.json', sm.describe_training_job(TrainingJobName=name))
    print(json.dumps({'job_name': name, 's3_results': 's3://' + BUCKET + '/' + prefix + '/results/'}))


if __name__ == '__main__':
    main()
