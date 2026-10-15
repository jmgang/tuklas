"""Verify the completed local repair, with fresh query encoding and exact ranking.

Read every saved vector checksum, recompute all rankings and metrics without
the runner's retrieval/scoring functions, and save NEW verification evidence.
No AWS, latency, deletion, overwritten reports or model changes.
"""
import argparse
import importlib.util
import json
from math import log2
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model2vec import StaticModel
from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.grade3 import select_cohorts, TOP10_METRICS, paired_with_teacher

spec = importlib.util.spec_from_file_location('repair_verify_source', ROOT/'scripts/run_ranking_512.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def load_vectors(directory, count, dimension):
    chunks, next_row = [], 0
    for path in sorted(directory.glob('*.npy')):
        record = runner.read(path.with_suffix('.json'))
        assert record['start'] == next_row
        assert runner.file_hash(path) == record['sha256']
        rows = np.load(path, allow_pickle=False)
        assert rows.dtype == np.float32 and rows.shape == (record['stop']-next_row, dimension)
        assert np.isfinite(rows).all()
        chunks.append(rows)
        next_row = record['stop']
    assert next_row == count
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    directory = ROOT/'runs'/args.run_id
    if (directory/'independent-verification.json').exists():
        raise FileExistsError('Verification already exists; preserve it')
    config = runner.read(directory/'settings.json')
    manifest = runner.read(directory/'manifest.json')
    source = runner.read(ROOT/'configs/disastir-qa-all-categories.json')
    corpus = load_corpus(ROOT/'data/raw/corpus.jsonl.gz', expected_sha256=config['corpus_sha256'])
    prepared = load_qa_collection(data_dir=ROOT/'data/raw', revision=source['revision'],
        category_checksums=source['categories'], seed=source['split_seed'], evaluation_fraction=source['evaluation_fraction'])
    mapped, _ = map_query_groups(prepared.evaluation, corpus)
    _, queries, _ = select_cohorts(prepared.evaluation, mapped, corpus)
    catalog_path = ROOT/config['verified_reference']/'shared-catalog.json'
    assert runner.file_hash(catalog_path) == manifest['shared_catalog_sha256']
    catalog = runner.read(catalog_path)
    assert [q.query_id for q in queries] == manifest['evaluation_query_ids'] == catalog['query_ids']
    train, validation, test = (set(manifest[key]) for key in ('training_query_ids','validation_query_ids','evaluation_query_ids'))
    assert not train & validation and not train & test and not validation & test
    assert (train | validation).issubset({q.query_id for q in prepared.development})
    assert not (train | validation) & {q.query_id for q in prepared.evaluation}
    assert runner.file_hash(ROOT/'scripts/run_ranking_512.py') == manifest['source_sha256']
    assert runner.file_hash(ROOT/('configs/disastir-512-improved-v1.json' if config.get('hard_negative_count') else 'configs/disastir-512-baseline-v1.json')) == manifest['config_sha256']
    selection = runner.read(directory/'selection.json')
    for filename, checksum in selection['model_files_sha256'].items():
        assert runner.file_hash(directory/'model'/filename) == checksum
    assert selection['epoch'] == max(selection['validation_history'],
        key=lambda x:(x['recall_at_10'],x['ndcg_at_10'],-x['epoch']))['epoch']
    owners = np.array([i for i,_,_ in catalog['segments']],dtype=np.int64)
    boundaries = np.r_[0,np.flatnonzero(owners[1:]!=owners[:-1])+1]
    assert len(boundaries) == len(corpus.ids) == 239704 and len(owners) == 240765
    reports, checks = {}, {}
    for name, path, prefix in [('potion_retrieval', ROOT/config['potion_baseline'], ''),
                              ('tokenlearn_retrieval_student', directory/'model', 'query: ')]:
        model = StaticModel.from_pretrained(str(path))
        vectors = load_vectors(directory/f'evaluation-vectors-{name}', len(owners), model.dim)
        report = runner.read(directory/f'quality-{name}.json')
        assert len(report['per_query']) == len(queries) == 1127
        for start in range(0,len(queries),32):
            batch = queries[start:start+32]
            qvectors = model.encode([prefix+q.query for q in batch],max_length=None,use_multiprocessing=False)
            segment_scores = qvectors @ vectors.T
            parent_scores = np.maximum.reduceat(segment_scores, boundaries, axis=1)
            for offset, (query, scores) in enumerate(zip(batch,parent_scores)):
                threshold = np.partition(scores,-10)[-10]
                candidates = np.flatnonzero(scores>=threshold)
                indices = sorted(candidates,key=lambda i:(-float(scores[i]),corpus.ids[i]))[:10]
                ids = [corpus.ids[i] for i in indices]
                row = report['per_query'][start+offset]
                assert row['query_id'] == query.query_id
                assert ids == [r['corpus_id'] for r in row['top10']]
                positions = [i+1 for i,key in enumerate(ids) if key in query.gold_ids]
                metrics = {f'recall_at_{k}':sum(p<=k for p in positions)/len(query.gold_ids) for k in (1,3,5,10)}
                metrics['mrr_at_10'] = 1/positions[0] if positions else 0
                ideal = sum(1/log2(i+1) for i in range(1,min(10,len(query.gold_ids))+1))
                metrics['ndcg_at_10'] = sum(1/log2(p+1) for p in positions)/ideal
                metrics['unjudged_at_10'] = sum(key not in query.judged_ids for key in ids)/10
                assert all(abs(metrics[k]-row['metrics'][k])<1e-12 for k in TOP10_METRICS)
            print(name, 'verified queries', min(start+32,len(queries)), flush=True)
        for key in TOP10_METRICS:
            assert abs(np.mean([r['metrics'][key] for r in report['per_query']])-report['summary']['overall'][key])<1e-12
        checks[name] = {'queries_independently_ranked':1127,'all_vector_checksums_verified':True,
                        'dimension':model.dim,'model_bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
                        'dtype':str(model.embedding.dtype),'vector_bytes':vectors.nbytes}
        reports[name]=report
        del vectors
    oyster = next(q for q in queries if q.query_id=='0291b8ce4f12580d907808480c1d1f64c3efb47d2a9fa9daec64ad3e85a04865')
    original = StaticModel.from_pretrained(str(ROOT/config['initial_student']))
    assert np.array_equal(original.embedding,np.load(directory/'weights-epoch-00.npy',allow_pickle=False))
    original_vectors = load_vectors(directory/'validation-vectors-epoch-00',len(owners),original.dim)
    oyster_original = runner.oyster_ranks(original,original_vectors,oyster,'query: ',owners,corpus)
    runner.save(directory/'oyster-original-student-diagnostic.json',oyster_original)
    paired = paired_with_teacher(reports['potion_retrieval']['per_query'],reports['tokenlearn_retrieval_student']['per_query'],
                                 seed=20261003,repetitions=10000,metrics=TOP10_METRICS)
    evidence = {'status':'passed','manifest_sha256':runner.file_hash(directory/'manifest.json'),
        'models':checks,'paired_student_minus_potion':paired,
        'training_evaluation_disjoint':True,'all_top10_unique_parent_ids':True,'original_oyster_ranks':oyster_original,
        'limitations':['Post-hoc development-trained repair, not a newly blind held-out experiment.',
                       'Query bootstrap assumes independent queries; pooled judgments and topic dependence remain.']}
    runner.save(directory/'independent-verification.json',evidence)
    print(json.dumps({'status':'passed','models':checks,'paired_recall10':paired['metrics']['recall_at_10'],
                      'original_oyster_gold_ranks':[(r['corpus_id'],r['rank']) for r in oyster_original['gold_ranks']]},indent=2))


if __name__ == '__main__':
    main()

