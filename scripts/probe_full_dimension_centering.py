"""Development-only, 1024-D static-vector centering diagnostic.

No PCA reduction, training, held-out scoring, S3 writes or prior-file changes.
The pre-SIF variant is a diagnostic approximation, not a claim that the saved
TokenLearn vectors exactly equal a fresh Model2Vec `pca_dims="auto"` run.
"""

import argparse
import gc
import json
from pathlib import Path

import boto3
import numpy as np
from model2vec import StaticModel
from safetensors.numpy import load as load_safetensors
from tokenizers import Tokenizer

from probe_ranked_interpolation import BUCKET, JOB, PREFIX, read_verified
from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score
from pycon_model2vec_demo.multiwikiqa_ranking_v3 import prepare


def metrics(model, validation, chunks, owners, ids):
    documents = model.encode([c['text'] for c in chunks], batch_size=512,
                             max_length=None, use_multiprocessing=False)
    questions = model.encode([q['question'] for q in validation], batch_size=512,
                             max_length=None, use_multiprocessing=False)
    rows = []
    for start in range(0, len(validation), 64):
        rankings = rank_parents(questions[start:start + 64], documents, owners, ids)
        for q, ranking in zip(validation[start:start + 64], rankings):
            rows.append({'query_id': q['query_id'], 'language': q['language'],
                         'gold_parent_id': q['gold_parent_id'],
                         'metrics': score(q, ranking, chunks),
                         'top10_cosine': [float(r[2]) for r in ranking]})
    summary = {}
    for language in ('all', 'en', 'tl'):
        subset = rows if language == 'all' else [r for r in rows if r['language'] == language]
        summary[language] = {'questions': len(subset),
                             'hits@10': int(sum(r['metrics']['Recall@10'] for r in subset)),
                             'nDCG@10': float(np.mean([r['metrics']['nDCG@10'] for r in subset])),
                             'MRR@10': float(np.mean([r['metrics']['MRR@10'] for r in subset])),
                             'mean_top1_cosine': float(np.mean([r['top10_cosine'][0] for r in subset])),
                             'mean_top10_cosine': float(np.mean([r['top10_cosine'][-1] for r in subset]))}
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'runs' / JOB / 'mixed-plan.json').read_text())
    s3 = boto3.Session(profile_name='admin-general', region_name='ap-southeast-1').client('s3')
    layout_raw, _ = read_verified(s3, plan['evaluation_layout']['key'],
                                  plan['evaluation_layout']['sha256'], plan['evaluation_layout']['bytes'])
    layout = json.loads(layout_raw)
    query_raw, _ = read_verified(s3, plan['annotations']['key'],
                                 plan['annotations']['sha256'], plan['annotations']['bytes'])
    queries = [json.loads(line) for line in query_raw.splitlines()]
    train, validation, skipped = prepare(queries, layout['chunks'])
    if (len(train), len(validation), len(skipped)) != (1734, 338, 6):
        raise ValueError('Development split changed')
    del query_raw, queries, train
    chunks, ids = layout['chunks'], layout['parents']
    positions = {name: i for i, name in enumerate(ids)}
    owners = [positions[chunk['parent_id']] for chunk in chunks]
    raw, _ = read_verified(s3, PREFIX + '/models/approved_articles_ranked/epoch-00/tokenizer.json')
    tokenizer = Tokenizer.from_str(raw.decode())
    raw, _ = read_verified(s3, PREFIX + '/models/approved_articles_ranked/epoch-00/config.json')
    config = json.loads(raw)
    del raw
    if config.get('normalize') is not True:
        raise ValueError('Unexpected static model configuration')
    results = []
    for epoch in (0, 1):
        raw, digest = read_verified(s3,
            PREFIX + f'/models/approved_articles_ranked/epoch-{epoch:02d}/model.safetensors')
        tensor = load_safetensors(raw)
        if len(tensor) != 1:
            raise ValueError('Unexpected weight structure')
        weight = np.array(next(iter(tensor.values())), dtype=np.float32, copy=True)
        del raw, tensor
        gc.collect()
        if weight.shape != (249999, 1024) or not np.isfinite(weight).all():
            raise ValueError('Weights changed size or are nonfinite')
        print(f'Verified epoch {epoch} weights, SHA256 {digest}', flush=True)
        n = len(weight)
        inv_rank = 1 / np.arange(2, n + 2, dtype=np.float64)
        probability = inv_rank / inv_rank.sum()
        sif = (1e-4 / (1e-4 + probability)).astype(np.float32)
        variants = {
            'unchanged': weight,
            'center_weighted_vectors': weight - weight.mean(axis=0, dtype=np.float64).astype(np.float32),
            'estimate_pre_sif_centering': weight - sif[:, None] *
                (weight / sif[:, None]).mean(axis=0, dtype=np.float64).astype(np.float32),
        }
        del inv_rank, probability, sif
        for name, vectors in variants.items():
            model = StaticModel(vectors=vectors, tokenizer=Tokenizer.from_str(tokenizer.to_str()),
                                config=config, normalize=True)
            summary, rows = metrics(model, validation, chunks, owners, ids)
            results.append({'epoch': epoch, 'variant': name, 'summary': summary,
                            'per_query': rows, 'source_weights_sha256': digest})
            print(f'epoch={epoch} variant={name} summary={summary}', flush=True)
            del model
            gc.collect()
        del variants, weight
        gc.collect()
    baseline = next(r for r in results if r['epoch'] == 0 and r['variant'] == 'unchanged')
    if abs(baseline['summary']['all']['nDCG@10'] - .642052976562431) > 1e-8:
        raise ValueError('Epoch-0 metric failed exact historical reproduction')
    for result in results:
        original = next(r for r in results if r['epoch'] == 0 and r['variant'] == result['variant'])
        result['versus_same_variant_epoch0'] = {
            language: {
                'gained': [current['query_id'] for before, current in zip(
                    original['per_query'], result['per_query']) if current['language'] == language and
                    before['metrics']['Recall@10'] == 0 and current['metrics']['Recall@10'] == 1],
                'lost': [current['query_id'] for before, current in zip(
                    original['per_query'], result['per_query']) if current['language'] == language and
                    before['metrics']['Recall@10'] == 1 and current['metrics']['Recall@10'] == 0],
            } for language in ('en', 'tl')}
    report = {'scope': 'development-only centering diagnostic; no training/heldout scoring',
              'source_job': JOB, 'full_dimensions': 1024, 'no_pca_reduction': True,
              'assumptions': ['unweighted simple token mean',
                              'estimated inverse-SIF variant uses Model2Vec coefficient 1e-4; '
                              'it is not a fresh official PCA-auto initialization'],
              'layout_sha256': plan['evaluation_layout']['sha256'],
              'annotation_sha256': plan['annotations']['sha256'],
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    print(f'Preserved diagnostic at {args.output}', flush=True)


if __name__ == '__main__':
    main()
