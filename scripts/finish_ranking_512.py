"""Serial 512-capacity baseline and improvement candidate, preserving all outputs."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pycon_model2vec_demo.grade3 import paired_with_teacher, TOP10_METRICS


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder-pid', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('One new folder name required')
    directory = ROOT/'runs'/args.run_id
    directory.mkdir(exist_ok=False)
    def event(stage, **values):
        record = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, **values)
        save(directory/'events'/f'{uuid4().hex}.json', record)
        print(json.dumps(record), flush=True)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'TOKENIZERS_PARALLELISM':'false',
           'HF_HOME':str(ROOT/'models/.hf-cache'), 'HF_HUB_OFFLINE':'1',
           'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'VECLIB_MAXIMUM_THREADS':'4'}
    candidates = [('baseline', 'disastir-512-baseline-20261003-v1'),
                  ('improved', 'disastir-512-improved-20261003-v1')]
    pinned = {str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest()
              for path in [ROOT/'scripts/run_ranking_512.py',ROOT/'scripts/verify_ranking_512.py',
                           ROOT/'configs/disastir-512-baseline-v1.json', ROOT/'configs/disastir-512-improved-v1.json',
                           ROOT/'src/pycon_model2vec_demo/ranking_512.py']}
    save(directory/'source-pins.json', pinned)
    completion = ROOT/'runs/disastir-512-rebuild-20261003-v2/completion.json'
    event('waiting-for-actual-512-tokenlearn-completion', builder_pid=args.builder_pid)
    while not completion.exists():
        try:
            os.kill(args.builder_pid, 0)
        except ProcessLookupError:
            event('blocked-builder-stopped-without-completion')
            raise RuntimeError('Builder stopped; no automatic restart or reuse of partial artifacts')
        time.sleep(15)
    if read(completion)['dimension'] != 512:
        raise ValueError('Builder did not produce 512 dimensions')
    for path, digest in pinned.items():
        if sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Source or configuration changed after schedule freeze')
    results = {}
    for name, run in candidates:
        for stage, command in [
            ('ranking-and-fresh-quality', [sys.executable, '-u', 'scripts/run_ranking_512.py',
                '--run-id', run, '--config', f'configs/disastir-512-{name}-v1.json']),
            ('independent-verification', [sys.executable, '-u', 'scripts/verify_ranking_512.py', '--run-id', run])]:
            event(stage, candidate=name, run_id=run)
            with (directory/f'{name}-{stage}.log').open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                save(directory/f'{name}-{stage}-process.json', dict(pid=process.pid, command=command))
                status = process.wait()
            if status:
                event('failed', candidate=name, operation=stage, exit_code=status)
                raise RuntimeError('Candidate failed; preserve output and diagnose before repair')
        folder = ROOT/'runs'/run
        results[name] = dict(selection=read(folder/'selection.json'),
                             comparison=read(folder/'comparison.json'),
                             verification=read(folder/'independent-verification.json'), run_id=run)
        event('candidate-complete-and-verified', candidate=name,
              recall_at_10=results[name]['comparison']['models']['tokenlearn_retrieval_student']['overall']['recall_at_10'])
    baseline = ROOT/'runs'/candidates[0][1]
    improved = ROOT/'runs'/candidates[1][1]
    paired = paired_with_teacher(read(baseline/'quality-tokenlearn_retrieval_student.json')['per_query'],
                                 read(improved/'quality-tokenlearn_retrieval_student.json')['per_query'],
                                 seed=20261003, repetitions=10000, metrics=TOP10_METRICS)
    def validation_key(name):
        selected = results[name]['selection']
        row = next(x for x in selected['validation_history'] if x['epoch']==selected['epoch'])
        return row['recall_at_10'], row['ndcg_at_10']
    chosen = max(results, key=validation_key)
    save(directory/'comparison.json', dict(results=results, paired_improved_minus_baseline=paired,
        development_selected_candidate=chosen, no_latency_measured=True,
        limitations=['Post-hoc test, not a newly blind benchmark.', 'One training seed; query intervals do not measure training-seed uncertainty.',
                     'Improved candidate changes batching, negatives and schedule together; not a pure epoch ablation.',
                     'Domain adaptation, not identical Potion teacher/vocabulary/full-mixture reproduction.']))
    event('complete', development_selected_candidate=chosen)


if __name__ == '__main__':
    main()
