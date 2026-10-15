"""Expose historical deployed code without altering any original artifact."""
from pathlib import Path
from hashlib import sha256
import json
import tarfile
from urllib.parse import quote
import boto3

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'runs/aws-demo-source-index-20261004T132400Z/historical-ec2'
BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'
BASE = 'pycon-davao-model2vec/pycon-isolated-20261002-192911-eb3adc'


def plan():
    s3 = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1').client('s3')
    entries, contents = [], {}
    def add(group, name, raw, key, expected=None, member=None):
        actual = sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError('Historical source checksum mismatch: ' + name)
        if not raw.endswith(b'\n'):
            raise ValueError('Patch copy requires preserved final newline: ' + name)
        target = DEST / group / name
        contents[str(target)] = raw.decode('utf-8')
        entries.append({'local_path': str(target), 'source_s3': 's3://' + BUCKET + '/' + key,
            'archive_member': member, 'sha256': actual, 'bytes': len(raw),
            'aws_console': 'https://s3.console.aws.amazon.com/s3/object/' + BUCKET +
                '?region=ap-southeast-1&prefix=' + quote(key, safe='')})
    deployment = json.loads((ROOT / 'runs/pycon-isolated-20261002-192911-eb3adc/deployment-inputs.json').read_text())
    for archive_name in ('host-code.tar.gz', 'teacher-code.tar.gz'):
        local = ROOT / 'runs/pycon-isolated-20261002-192911-eb3adc' / archive_name
        key = BASE + '/inputs/' + archive_name
        remote = s3.get_object(Bucket=BUCKET, Key=key)['Body'].read()
        if sha256(remote).digest() != sha256(local.read_bytes()).digest():
            raise ValueError('Local/AWS archive mismatch: ' + archive_name)
        with tarfile.open(local) as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith('.py'):
                    continue
                if archive_name == 'teacher-code.tar.gz' and member.name != 'code/inference.py':
                    continue
                raw = archive.extractfile(member).read()
                expected = deployment['code_sha256'].get(member.name)
                if member.name == 'code/inference.py':
                    expected = deployment['code_sha256']['scripts/isolated_sagemaker_inference.py']
                if expected is None:
                    raise ValueError('No frozen source hash: ' + member.name)
                add('original-isolated', member.name, raw, key, expected, member.name)
    for run in ('pycon-continue-20261003-053452-4bc896', 'pycon-repair-20261003-065954-4b06a8'):
        settings = json.loads((ROOT / 'runs' / run / 'continuation-settings.json').read_text())
        for role in ('worker', 'controller'):
            record = settings['continuation_sources'][role]
            raw = s3.get_object(Bucket=BUCKET, Key=record['key'])['Body'].read()
            add(run, role + '.py', raw, record['key'], record['sha256'])
    manifest = {'scope': 'Original isolated DisastIR EC2 deployment, GPU endpoint handler, continuation and repaired workers/controllers',
        'region': 'ap-southeast-1', 'profile': 'admin-general', 'entries': entries,
        'all_originals_preserved': True, 'no_compute_launched': True}
    contents[str(DEST / 'manifest.json')] = json.dumps(manifest, indent=2) + '\n'
    lines = ['# Historical AWS evaluation source code', '',
        'Exact deployed source copies, checked against frozen launch hashes and original AWS objects. No instances were restarted.', '',
        'Original isolated deployment includes the retrieval/scoring modules, model HTTP service, GPU inference handler, benchmark worker and controller. Continuation and repair versions are retained separately.', '',
        '| Version / source | Local code | Original AWS object |', '| --- | --- | --- |']
    for e in entries:
        label = str(Path(e['local_path']).relative_to(DEST))
        lines.append('| ' + label + ' | [Open source](' + e['local_path'] + ') | [AWS source](' + e['aws_console'] + ') |')
    lines += ['', 'For archive-backed files, the AWS link opens the preserved deployment bundle; the local link opens the individual source file. SHA256 hashes and archive member names are recorded in [manifest.json](' + str(DEST / 'manifest.json') + ').', '',
        'The rejected initial single-host run is not represented as a valid result. Its runner source is preserved in the original deployment bundle as scripts/run_fresh_aws_benchmark.py.', '',
        'Current and earlier SageMaker training/evaluation sources: [SageMaker source index](' + str(DEST.parent / 'README-byte-exact.md') + ').']
    contents[str(DEST / 'README.md')] = '\n'.join(lines) + '\n'
    return s3, manifest, contents


if __name__ == '__main__':
    import sys
    s3, manifest, contents = plan()
    if '--verify-and-index' in sys.argv:
        for target, original in contents.items():
            if Path(target).read_bytes() != original.encode():
                raise ValueError('Demo copy differs: ' + target)
        key = 'pycon-davao-model2vec/demo-historical-source-index-20261004T134000Z/manifest.json'
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, IfNoneMatch='*',
            Metadata={'sha256': sha256(data).hexdigest()})
        print(json.dumps({'verified_files': len(manifest['entries']), 'manifest_s3': 's3://' + BUCKET + '/' + key}))
    else:
        patch = ['*** Begin Patch']
        for target, content in contents.items():
            if Path(target).exists():
                raise FileExistsError(target)
            patch += ['*** Add File: ' + target] + ['+' + line for line in content.split('\n')[:-1]]
        patch += ['*** End Patch']
        print(json.dumps({'patch': '\n'.join(patch), 'source_count': len(manifest['entries'])}))
