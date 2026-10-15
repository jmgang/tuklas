"""Launch one immutable, evaluation-only job for a development-selected student."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import boto3

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'


def digest(path):
    value = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with path.open('x') as target:
        json.dump(value, target, sort_keys=True, indent=2, default=str)
        target.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=('external_wikipedia', 'approved_plus_external'))
    parser.add_argument('source_job')
    args = parser.parse_args()
    arm = args.arm
    source_job = args.source_job
    source_prefix = 'pycon-davao-model2vec/' + source_job + '/results'
    session = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1')
    if session.client('sts').get_caller_identity()['Account'] != '<aws-account-id>':
        raise ValueError('Wrong AWS account')
    sm, s3 = session.client('sagemaker'), session.client('s3')
    if sm.describe_training_job(TrainingJobName=source_job)['TrainingJobStatus'] != 'Completed':
        raise ValueError('Development probe not complete')
    for match in ('pycon-multiwikiqa', 'pycon-sealion'):
        for page in sm.get_paginator('list_training_jobs').paginate(
                NameContains=match, StatusEquals='InProgress'):
            if page['TrainingJobSummaries']:
                raise ValueError('Another experimental SageMaker job is active')
    predecessor = json.loads((ROOT / 'runs' / source_job / 'mixed-plan.json').read_text())
    if predecessor['ranking_arm'] != arm:
        raise ValueError('Frozen source arm mismatch')
    response = s3.get_object(Bucket=BUCKET, Key=source_prefix + '/reports/teacher-margin-arm-completion.json')
    try:
        completion = json.loads(response['Body'].read())
    finally:
        response['Body'].close()
    if (completion['arm'] != arm or completion['heldout_scored']
            or completion['source_plan_sha256'] != sha256(
                (ROOT / 'runs' / source_job / 'mixed-plan.json').read_bytes()).hexdigest()):
        raise ValueError('Source development selection changed')
    from pycon_model2vec_demo.multiwikiqa_ranking_v5 import select_checkpoint
    selected_epoch = completion['training_report']['selected_epoch']
    if (selected_epoch != select_checkpoint(completion['training_report']['history'])
            or selected_epoch == 0):
        raise ValueError('No improving development checkpoint selected')
    artifacts = completion['artifacts']
    model = {}
    for filename in ('README.md', 'config.json', 'model.safetensors',
                     'modules.json', 'tokenizer.json'):
        relative = f'models/{arm}_ranked/epoch-{selected_epoch:02d}/' + filename
        spec = artifacts[relative]
        key = source_prefix + '/' + relative
        head = s3.head_object(Bucket=BUCKET, Key=key)
        if head['ContentLength'] != spec['bytes'] or head['Metadata'].get('sha256') != spec['sha256']:
            raise ValueError('Frozen model object changed: ' + filename)
        model[filename] = {'key': key, **spec}
    name = ('pycon-mqa-hold-' + ('external' if arm == 'external_wikipedia' else 'mixture') + '-' +
            datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:6])
    directory = ROOT / 'runs' / name
    directory.mkdir(exist_ok=False)
    prefix = 'pycon-davao-model2vec/' + name
    files = {
        'aws_multiwikiqa_v5_heldout_eval.py': ROOT / 'scripts/aws_multiwikiqa_v5_heldout_eval.py',
        'multiwikiqa_evaluation.py': ROOT / 'src/pycon_model2vec_demo/multiwikiqa_evaluation.py',
    }
    for path in files.values():
        compile(path.read_text(), str(path), 'exec')
    plan = {
        'purpose': 'heldout evaluation only; no training or checkpoint selection',
        'bucket': BUCKET, 'output_prefix': prefix + '/results',
        'source_job': source_job, 'selected_epoch': selected_epoch,
        'candidate_label': arm + '_teacher_margin_epoch' + str(selected_epoch),
        'baseline_label': arm,
        'layout': predecessor['evaluation_layout'],
        'annotations': predecessor['annotations'],
        'baseline_report': (predecessor['frozen_baseline_reports'][arm]
                            if arm != 'approved_plus_external'
                            else predecessor['recovered_quality'][arm]),
        'selected_model': model,
        'source_files': {filename: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                         for filename, path in files.items()},
        'frozen_protocol': {'parents': 1294, 'chunks': 6546, 'questions': 10000,
                            'heldout_questions': 7922, 'cutoff': 10,
                            'parent_reduction': 'maximum chunk cosine',
                            'quality_gate': 'No lost Recall@10 question and no aggregate retrieval metric decline in either language'},
    }
    save(directory / 'evaluation-plan.json', plan)
    files['evaluation-plan.json'] = directory / 'evaluation-plan.json'
    for filename, path in files.items():
        with path.open('rb') as source:
            s3.put_object(Bucket=BUCKET, Key=prefix + '/input/' + filename,
                          Body=source, IfNoneMatch='*',
                          Metadata={'sha256': digest(path)})
        head = s3.head_object(Bucket=BUCKET, Key=prefix + '/input/' + filename)
        if head['ContentLength'] != path.stat().st_size or head['Metadata'].get('sha256') != digest(path):
            raise ValueError('Staged source changed')
    request = json.loads((ROOT / 'runs' / source_job / 'create-training-job-request.json').read_text())
    request['TrainingJobName'] = name
    request['AlgorithmSpecification']['ContainerArguments'] = [
        'python /opt/ml/input/data/input/aws_multiwikiqa_v5_heldout_eval.py']
    request['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri'] = (
        's3://' + BUCKET + '/' + prefix + '/input/')
    request['OutputDataConfig']['S3OutputPath'] = 's3://' + BUCKET + '/' + prefix + '/native-output/'
    request['StoppingCondition']['MaxRuntimeInSeconds'] = 1800
    request['Environment']['SEALION_OUTPUT_PREFIX'] = prefix + '/results'
    request['Tags'] = [
        {'Key': 'Project', 'Value': 'pycon-davao-model2vec-demo'},
        {'Key': 'Purpose', 'Value': 'frozen-v6-heldout-' + arm},
        {'Key': 'DemoSourceManifest', 'Value': 's3://' + BUCKET + '/' + prefix + '/input/evaluation-plan.json'},
    ]
    save(directory / 'create-training-job-request.json', request)
    save(directory / 'resource-record.json', {
        'job_name': name, 'region': 'ap-southeast-1', 'profile': 'admin-general',
        's3_results': 's3://' + BUCKET + '/' + prefix + '/results/',
        'instance': 'ml.g4dn.2xlarge', 'max_runtime_seconds': 1800,
        'rate_usd_per_hour': 1.315, 'max_runtime_compute_estimate_usd': 0.6575,
        'credits_verified': False, 'no_endpoint': True,
        'managed_release_on_terminal': True, 'training': False,
        'source_job': source_job, 'selected_epoch': selected_epoch})
    save(directory / 'launch-response.json', sm.create_training_job(**request))
    save(directory / 'initial-job-state.json', sm.describe_training_job(TrainingJobName=name))
    print(json.dumps({'job': name, 'results': 's3://' + BUCKET + '/' + prefix + '/results/'}))


if __name__ == '__main__':
    main()
