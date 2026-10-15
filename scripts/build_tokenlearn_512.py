"""Fresh 512-dimensional Model2Vec + real TokenLearn rebuild.

Reuses only checksummed 768-dimensional teacher targets, not 256-dimensional
student weights or retrieval scores. No queries/qrels, AWS or network encoding.
Every output path is new; nothing is deleted or replaced.
"""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / 'models/.hf-cache')
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill import distill
from sklearn.decomposition import PCA
import tokenlearn.train as tokenlearn_train
from tokenlearn.utils import collect_means_and_texts
from pycon_model2vec_demo.distillation import file_hashes
from pycon_model2vec_demo.data import _file_sha256
from pycon_model2vec_demo.tokenlearn_checkpoint import independent_tokenlearn_checkpoints


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path):
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def event(directory, stage, **values):
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **values}
    save(directory / 'events' / f'{uuid4().hex}.json', record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('Run ID must be one new folder name')
    directory = ROOT / 'runs' / args.run_id
    directory.mkdir(exist_ok=False)
    config_path = ROOT / 'configs/disastir-512-rebuild-v1.json'
    config = read(config_path)
    save(directory / 'settings.json', config)
    if version('tokenlearn') != config['tokenlearn_version']:
        raise ValueError('Unexpected TokenLearn version')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['shuffle_seed'])
    np.random.seed(config['shuffle_seed'])
    event(directory, 'checking-reusable-teacher-targets')
    features = ROOT / config['feature_directory']
    manifest = read(features / 'manifest.json')
    old_config = read(ROOT / 'configs/models-v1.json')
    original = read(ROOT / 'runs/plain-e5-f52bf8ec8c71-provenance.json')
    if file_hashes(ROOT / config['teacher']) != original['teacher']['files_sha256']:
        raise ValueError('Teacher changed')
    hashes = file_hashes(features)
    hashes.pop('manifest.json', None)
    if hashes != manifest['feature_files_sha256']:
        raise ValueError('Teacher target files changed')
    data = read(ROOT / 'configs/disastir-qa-all-categories.json')
    if manifest['corpus_sha256'] != data['corpus_sha256']:
        raise ValueError('Wrong teacher-target corpus')
    if _file_sha256(ROOT / 'data/raw/corpus.jsonl.gz') != data['corpus_sha256']:
        raise ValueError('Corpus changed')
    vocab_record = read(ROOT / config['vocabulary_record'])
    canonical = json.dumps(vocab_record['requested_terms'], ensure_ascii=False,
                           separators=(',', ':')).encode()
    if sha256(canonical).hexdigest() != vocab_record['requested_terms_sha256']:
        raise ValueError('Vocabulary selection changed')
    vocabulary = [item['term'] for item in vocab_record['requested_terms']]
    event(directory, 'fresh-model2vec-distillation', dimension=512, dtype='float32')
    base = distill(model_name=str(ROOT / config['teacher']), vocabulary=vocabulary,
                   device=config['device'], pca_dims=512, quantize_to='float32',
                   sif_coefficient=old_config['students']['plain']['sif_coefficient'],
                   token_remove_pattern=old_config['students']['plain']['token_remove_pattern'],
                   trust_remote_code=False, vocabulary_quantization=None, pooling='mean')
    previous = StaticModel.from_pretrained(str(ROOT / 'models/custom-e5-f52bf8ec8c71'))
    if base.dim != 512 or base.tokens != previous.tokens:
        raise ValueError('New dimension or unchanged-vocabulary guard failed')
    base.save_pretrained(directory / 'distilled-model')
    del previous
    gc.collect()
    texts, targets = collect_means_and_texts(sorted(features.glob('feature_*.json')))
    if targets.shape != (config['expected_rows'], 768) or len(texts) != config['expected_rows']:
        raise ValueError('Missing, filtered or misaligned target rows')
    if not np.isfinite(targets).all():
        raise ValueError('Invalid teacher targets')
    order = np.random.default_rng(config['shuffle_seed']).permutation(len(texts))
    texts = [texts[int(i)] for i in order]
    targets = targets[order]
    save(directory / 'provenance.json', {
        'teacher': original['teacher'], 'features_manifest_sha256': file_hash(features / 'manifest.json'),
        'corpus_sha256': data['corpus_sha256'], 'rows': len(texts),
        'config_sha256': file_hash(config_path), 'source_sha256': file_hash(Path(__file__)),
        'checkpoint_fix_sha256': file_hash(ROOT / 'src/pycon_model2vec_demo/tokenlearn_checkpoint.py'),
        'dimension': 512, 'vocabulary_entries': len(base.tokens),
        'vocabulary_unchanged': True, 'teacher_target_dimension': 768,
        'no_query_or_qrel_training': True, 'evaluation_dataset_change': False,
        'recipe': 'E5/DisastIR domain adaptation; not identical full Potion pretraining/retrieval mixture',
    })

    class AuditedPCA(PCA):
        def fit_transform(self, x, y=None):
            result = super().fit_transform(x, y)
            cumulative = np.cumsum(self.explained_variance_ratio_)
            save(directory / 'pca-audit.json', {
                'input_dimension': x.shape[1], 'output_dimension': result.shape[1],
                'explained_variance_at_256': float(cumulative[255]),
                'explained_variance_at_512': float(cumulative[511]),
                'note': 'Variance retention is not a retrieval-quality score.',
            })
            with (directory / 'pca-components.npz').open('xb') as stream:
                np.savez(stream, components=self.components_, mean=self.mean_,
                         variance_ratio=self.explained_variance_ratio_)
            event(directory, 'target-pca-complete', retained_variance_512=float(cumulative[511]))
            return result

    class EpochRecorder(logging.Handler):
        def emit(self, record):
            if record.name == 'tokenlearn.pretrain':
                event(directory, 'tokenlearn-training', message=record.getMessage())

    handler = EpochRecorder()
    logger = logging.getLogger('tokenlearn.pretrain')
    logger.addHandler(handler)
    original_pca = tokenlearn_train.PCA
    tokenlearn_train.PCA = AuditedPCA
    try:
        event(directory, 'tokenlearn-started', dimension=512, passage_count=len(texts))
        with independent_tokenlearn_checkpoints():
            trained = tokenlearn_train.train_model(base, texts, targets, device=config['device'], pca_dims=512)
    finally:
        tokenlearn_train.PCA = original_pca
        logger.removeHandler(handler)
    if trained.dim != 512 or trained.tokens != base.tokens or not np.isfinite(trained.embedding).all():
        raise ValueError('Invalid TokenLearn output')
    trained.save_pretrained(directory / 'model')
    reloaded = StaticModel.from_pretrained(str(directory / 'model'))
    probe = ['query: Where is an evacuation shelter?', 'passage: A shelter opens after flooding.']
    if not np.allclose(trained.encode(probe), reloaded.encode(probe), atol=2e-6):
        raise ValueError('TokenLearn save/load parity failed')
    save(directory / 'completion.json', {
        'status': '512-dimensional-TokenLearn-student-created-retrieval-training-and-evaluation-pending',
        'dimension': trained.dim, 'vocabulary_entries': len(trained.tokens),
        'model_files_sha256': file_hashes(directory / 'model'),
        'quality_measured': False, 'latency_measured': False,
    })
    event(directory, '512-tokenlearn-build-complete', quality_measured=False)


if __name__ == '__main__':
    main()
