"""Complete, replayable quality-only scoring. No timing or model selection."""
import math
import numpy as np
from multiwikiqa_evaluation import rank_parents, score

METRICS = tuple([f'Recall@{k}' for k in (1, 3, 5, 10)] +
                ['MRR@10', 'nDCG@10'] +
                [f'AnswerSpanCoverage@{k}' for k in (1, 3, 5, 10)])


def aggregate(rows):
    if not rows or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate question records')
    for row in rows:
        if set(row['metrics']) != set(METRICS):
            raise ValueError('Incomplete metric schema')
        for value in row['metrics'].values():
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Invalid metric value')
    result = {'questions': len(rows), 'denominators': {}}
    for key in METRICS:
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result['denominators'][key] = len(values)
    return result


def evaluate(documents, queries, cohort, owners, parents, chunks):
    if len(queries) != len(cohort):
        raise ValueError('Query-vector alignment mismatch')
    rows = []
    for start in range(0, len(cohort), 64):
        rankings = rank_parents(queries[start:start+64], documents, owners, parents)
        for query, ranking in zip(cohort[start:start+64], rankings, strict=True):
            if len(ranking) != 10 or len({r[0] for r in ranking}) != 10:
                raise ValueError('Expected exactly ten unique parent results')
            rows.append({'query_id': query['query_id'], 'language': query['language'],
                         'gold_parent_id': query['gold_parent_id'],
                         'partition': query['partition'], 'ranking': ranking,
                         'metrics': score(query, ranking, chunks)})
    return aggregate(rows), rows


def legacy_unweighted_encoder(model, pad_lookup):
    """Reproduce the historical TSS mean, resolving the large vocab only once."""
    from multiwikiqa_compact_retrieval import strip_padding_and_unknown
    weights = np.asarray(model.embedding, dtype=np.float32)
    tokenizer = model.tokenizer
    pad_id = pad_lookup(tokenizer)
    unknown_id = model.unk_token_id

    def encode(values, query=False):
        out = np.zeros((len(values), weights.shape[1]), dtype=np.float32)
        for start in range(0, len(values), 2048):
            batch = tokenizer.encode_batch_fast(values[start:start+2048], add_special_tokens=False)
            for offset, row in enumerate(batch):
                ids = strip_padding_and_unknown(row.ids, pad_id=pad_id, unknown_id=unknown_id)
                if ids:
                    out[start+offset] = weights[ids].mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    return encode
