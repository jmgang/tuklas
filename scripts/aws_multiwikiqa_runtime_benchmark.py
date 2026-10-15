"""Isolated local-runtime benchmark for final bilingual retrieval models.

Each model runs in a fresh child process. Model files, corpus vectors and exact
search remain on the same managed SageMaker host. Static models use CPU;
SEA-LION uses the local T4 GPU. Network/model download time is excluded.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from uuid import uuid4

INPUT = Path('/opt/ml/input/data/input')
WORK = Path('/opt/ml/runtime-benchmark')
BUCKET = 'sagemaker-ap-southeast-1-<aws-account-id>'


def digest(path):
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def percentile(values, point):
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=np.float64), point))


def child(model_name):
    import numpy as np
    import psutil
    import torch

    from multiwikiqa_evaluation import normalized

    plan = json.loads((WORK / 'benchmark-plan.json').read_text())
    layout = json.loads((WORK / 'shared-layout.json').read_text())
    queries = json.loads((WORK / 'timing-queries.json').read_text())
    spec = plan['models'][model_name]
    process = psutil.Process()
    torch.set_num_threads(plan['protocol']['cpu_threads'])
    np.random.seed(42)
    peak = {'rss': process.memory_info().rss}
    stopped = threading.Event()

    def watch():
        while not stopped.wait(.005):
            try:
                peak['rss'] = max(peak['rss'], process.memory_info().rss)
            except psutil.Error:
                return

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    baseline_rss = process.memory_info().rss
    if spec['runtime'] == 'gpu-transformer':
        from sentence_transformers import SentenceTransformer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        model = SentenceTransformer(str(WORK / 'models' / model_name), device='cuda',
                                    trust_remote_code=False)
        model.max_seq_length = 512
        torch.cuda.synchronize()
        load_ms = (time.perf_counter_ns() - started) / 1e6
    else:
        from model2vec import StaticModel
        started = time.perf_counter_ns()
        model = StaticModel.from_pretrained(str(WORK / 'models' / model_name))
        load_ms = (time.perf_counter_ns() - started) / 1e6
    loaded_rss = process.memory_info().rss

    texts = [chunk['text'] for chunk in layout['chunks']]
    started = time.perf_counter_ns()
    if spec['runtime'] == 'gpu-transformer':
        documents = model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, prompt='', show_progress_bar=False)
        torch.cuda.synchronize()
    else:
        documents = model.encode(texts, batch_size=512, max_length=None,
                                 use_multiprocessing=False)
    corpus_ms = (time.perf_counter_ns() - started) / 1e6
    documents = normalized(np.asarray(documents, dtype=np.float32))
    corpus_rss = process.memory_info().rss
    owners = np.asarray([layout['parents'].index(chunk['parent_id'])
                         for chunk in layout['chunks']], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
    if len(starts) != 1294:
        raise ValueError('Parent layout changed')

    def encode_one(text):
        if spec['runtime'] == 'gpu-transformer':
            result = model.encode([text], batch_size=1, normalize_embeddings=True,
                                  convert_to_numpy=True,
                                  prompt=plan['teacher']['query_prompt'],
                                  show_progress_bar=False)
            torch.cuda.synchronize()
        else:
            result = model.encode([text], batch_size=1, max_length=None,
                                  use_multiprocessing=False)
        return normalized(np.asarray(result, dtype=np.float32))[0]

    def search(vector):
        scores = documents @ vector
        parent_scores = np.maximum.reduceat(scores, starts)
        top = np.argpartition(-parent_scores, 9)[:10]
        return top[np.lexsort((np.asarray(layout['parents'])[top], -parent_scores[top]))]

    for text in [q['question'] for q in queries[:8]]:
        search(encode_one(text))
    encode_times, search_times, end_times = [], [], []
    returned = 0
    for _ in range(plan['protocol']['repetitions']):
        for row in queries:
            total_start = time.perf_counter_ns()
            encode_start = total_start
            vector = encode_one(row['question'])
            encode_end = time.perf_counter_ns()
            search_start = encode_end
            top = search(vector)
            search_end = time.perf_counter_ns()
            returned += len(top)
            encode_times.append((encode_end - encode_start) / 1e6)
            search_times.append((search_end - search_start) / 1e6)
            end_times.append((search_end - total_start) / 1e6)
    batch_times = []
    batch_texts = [q['question'] for q in queries]
    for _ in range(5):
        started = time.perf_counter_ns()
        if spec['runtime'] == 'gpu-transformer':
            model.encode(batch_texts, batch_size=32, normalize_embeddings=True,
                         convert_to_numpy=True, prompt=plan['teacher']['query_prompt'],
                         show_progress_bar=False)
            torch.cuda.synchronize()
        else:
            model.encode(batch_texts, batch_size=32, max_length=None,
                         use_multiprocessing=False)
        batch_times.append((time.perf_counter_ns() - started) / 1e9)
    stopped.set()
    monitor.join(timeout=1)
    peak['rss'] = max(peak['rss'], process.memory_info().rss)
    gpu = None
    if spec['runtime'] == 'gpu-transformer':
        gpu = {'device': torch.cuda.get_device_name(0),
               'allocated_after_load_bytes': int(torch.cuda.memory_allocated()),
               'reserved_after_load_bytes': int(torch.cuda.memory_reserved()),
               'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
               'peak_reserved_bytes': int(torch.cuda.max_memory_reserved())}
    def distribution(values):
        return {'mean_ms': float(np.mean(values)), 'p50_ms': percentile(values, 50),
                'p95_ms': percentile(values, 95), 'p99_ms': percentile(values, 99),
                'minimum_ms': float(np.min(values)), 'maximum_ms': float(np.max(values)),
                'samples': len(values)}
    result = {
        'model': model_name, 'runtime': spec['runtime'],
        'dimensions': int(documents.shape[1]),
        'model_weights_bytes': spec['model_weights_bytes'],
        'artifact_bytes': sum(path.stat().st_size for path in (WORK / 'models' / model_name).rglob('*')
                              if path.is_file()),
        'passage_vector_bytes': int(documents.nbytes),
        'load_time_ms': load_ms,
        'corpus_build': {'passages': len(documents), 'milliseconds': corpus_ms,
                         'passages_per_second': len(documents) / (corpus_ms / 1000)},
        'single_query_encode': distribution(encode_times),
        'exact_search': distribution(search_times),
        'end_to_end': distribution(end_times),
        'batch32_encoding': {'runs': 5, 'mean_seconds': float(np.mean(batch_times)),
                             'queries_per_second': 32 / float(np.mean(batch_times))},
        'memory': {'baseline_rss_bytes': baseline_rss,
                   'rss_after_model_load_bytes': loaded_rss,
                   'rss_after_corpus_vectors_bytes': corpus_rss,
                   'peak_rss_bytes': peak['rss'],
                   'peak_increment_over_baseline_bytes': peak['rss'] - baseline_rss,
                   'gpu': gpu},
        'returned_parent_ids': returned,
    }
    target = WORK / 'child-results' / (model_name + '.json')
    target.parent.mkdir(exist_ok=True)
    with target.open('x') as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write('\n')


def coordinator():
    import boto3
    import psutil
    from huggingface_hub import snapshot_download

    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'model2vec==0.9.0', 'sentence-transformers==5.7.0',
                    'transformers==5.3.0', 'numpy==2.5.3', 'psutil==7.1.3'],
                   check=True)
    plan = json.loads((INPUT / 'benchmark-plan.json').read_text())
    for filename, spec in plan['source_files'].items():
        path = INPUT / filename
        if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError('Staged source changed: ' + filename)
    WORK.mkdir(exist_ok=False)
    (WORK / 'events').mkdir(exist_ok=False)
    (WORK / 'models').mkdir(exist_ok=False)
    (WORK / 'benchmark-plan.json').write_bytes((INPUT / 'benchmark-plan.json').read_bytes())
    (WORK / 'multiwikiqa_evaluation.py').write_bytes((INPUT / 'multiwikiqa_evaluation.py').read_bytes())
    sys.path.insert(0, str(WORK))
    s3 = boto3.client('s3', region_name='ap-southeast-1')
    index = {}

    def publish(path, relative):
        spec = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        with path.open('rb') as source:
            s3.put_object(Bucket=BUCKET, Key=plan['output_prefix'] + '/' + relative,
                          Body=source, ContentLength=spec['bytes'], IfNoneMatch='*',
                          Metadata={'sha256': spec['sha256']})
        index[relative] = spec

    def save(relative, value):
        path = WORK / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as output:
            json.dump(value, output, sort_keys=True, indent=2)
            output.write('\n')
        publish(path, relative)

    def event(stage, **details):
        row = {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **details}
        save('events/' + uuid4().hex + '.json', row)
        print(json.dumps(row), flush=True)

    def get(spec, target):
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
            raise ValueError('Frozen object changed: ' + spec['key'])

    get(plan['layout'], WORK / 'shared-layout.json')
    annotations = WORK / 'annotations.jsonl'
    get(plan['annotations'], annotations)
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    timing = []
    for language in ('en', 'tl'):
        eligible = sorted((q for q in rows if q['partition'] == 'development'
                           and q['language'] == language), key=lambda q: q['query_id'])
        timing.extend(eligible[:16])
    save('timing-queries.json', timing)
    (WORK / 'timing-queries.json').write_text((WORK / 'timing-queries.json').read_text())
    for name, spec in plan['models'].items():
        target = WORK / 'models' / name
        if spec['source'] == 's3':
            target.mkdir(exist_ok=False)
            for filename, object_spec in spec['files'].items():
                get(object_spec, target / filename)
        else:
            snapshot_download(spec['repo_id'], revision=spec['revision'],
                              local_dir=target, local_dir_use_symlinks=False)
            for filename, expected in spec.get('verified_files', {}).items():
                path = target / filename
                if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                    raise ValueError('Pinned Hugging Face artifact changed: ' + name + '/' + filename)
        event('model-staged', model=name)
    results = {}
    for name in plan['models']:
        event('model-benchmark-start', model=name)
        environment = dict(os.environ)
        environment.update({'PYTHONPATH': str(WORK), 'OMP_NUM_THREADS': '8',
                            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                            'TOKENIZERS_PARALLELISM': 'false'})
        subprocess.run([sys.executable, str(INPUT / 'aws_multiwikiqa_runtime_benchmark.py'),
                        '--child', name], check=True, env=environment)
        result = json.loads((WORK / 'child-results' / (name + '.json')).read_text())
        results[name] = result
        publish(WORK / 'child-results' / (name + '.json'), 'reports/' + name + '.json')
        event('model-benchmark-complete', model=name,
              p50_ms=result['end_to_end']['p50_ms'],
              peak_rss_bytes=result['memory']['peak_rss_bytes'])
    hardware = {'instance_type': plan['instance_type'], 'platform': platform.platform(),
                'logical_cpu_count': psutil.cpu_count(),
                'physical_cpu_count': psutil.cpu_count(logical=False),
                'total_ram_bytes': psutil.virtual_memory().total}
    combined = {'status': 'complete', 'protocol': plan['protocol'], 'hardware': hardware,
                'models': results, 'artifacts': dict(index),
                'source_plan_sha256': digest(INPUT / 'benchmark-plan.json'),
                'limitations': ['Static models run on CPU; teacher runs on local GPU, so inference hardware differs',
                                'Exact search over 6546 local passage vectors; no ANN or network hop',
                                'Single cold load measurement; 160 warmed single-query samples per model',
                                'RSS includes Python runtime and imported native libraries',
                                'Latency is system-specific and not retrieval-quality evidence']}
    save('reports/runtime-benchmark.json', combined)
    save('reports/completion.json', {'status': 'complete', 'models': list(results),
                                     'artifacts': dict(index),
                                     'compute_release': 'managed training compute releases on exit'})
    event('complete', models=len(results))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        child(sys.argv[2])
    elif len(sys.argv) == 1:
        coordinator()
    else:
        raise SystemExit('Usage: script [--child MODEL]')
