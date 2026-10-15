"""512-candidate helpers; labels are supplied from development only."""
import numpy as np


def _conflicts(row, other):
    docs = {row['positive_id'], row['negative_id']}
    other_docs = {other['positive_id'], other['negative_id']}
    return (row['query_id'] == other['query_id'] or bool(docs & other_docs)
            or bool(docs & other['positive_ids']) or bool(other_docs & row['positive_ids']))


def packed_safe_batches(rows, size):
    """First-fit packing retains every query, including singleton batches.

    One MNRL singleton still has its explicit judged-negative candidate.
    Known positives at every relevance grade are protected. Unknown cross-query
    positives remain possible and must be disclosed, not relabeled as irrelevant.
    """
    if size < 1:
        raise ValueError('Batch size must be positive')
    batches = []
    for row in rows:
        docs = {row['positive_id'], row['negative_id']}
        if len(docs) != 2 or row['negative_id'] in row['positive_ids']:
            raise ValueError('Negative is duplicate or known relevant')
        for batch in batches:
            if len(batch) < size and not any(_conflicts(row, other) for other in batch):
                batch.append(row)
                break
        else:
            batches.append([row])
    yield from batches


def hardest_judged_negatives(query, ids, vectors, count):
    """Rank ONLY the provided grade-zero candidate pool; stable ID tie break."""
    vectors = np.asarray(vectors)
    query = np.asarray(query)
    if (vectors.ndim != 2 or query.ndim != 1 or vectors.shape != (len(ids), len(query))
            or not ids or len(set(ids)) != len(ids) or count < 1
            or not np.isfinite(vectors).all() or not np.isfinite(query).all()):
        raise ValueError('Invalid negative pool')
    scores = vectors @ query
    return [ids[index] for index in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:count]]
