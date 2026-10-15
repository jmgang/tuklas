"""Leakage guards and independent checkpoint copies for retrieval adaptation.

The losses themselves come from Sentence Transformers, as in Potion's
published retrieval recipe. No test question is eligible for training.
"""
from hashlib import sha256


def independent_state(module):
    """Unlike state_dict alone, retain a real snapshot of earlier weights."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def split_development(groups, forbidden_ids, *, seed="disastir-ranking-repair-20261003", validation_fraction=.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    train, validation = [], []
    seen = set()
    for group in groups:
        if group.query_id in forbidden_ids or group.query_id in seen:
            raise ValueError("Evaluation or duplicate query in development training source")
        seen.add(group.query_id)
        number = int.from_bytes(sha256(f"{seed}\0{group.query_id}".encode()).digest()[:8], "big") / 2**64
        (validation if number < validation_fraction else train).append(group)
    if not train or not validation:
        raise ValueError("Empty training or validation partition")
    return train, validation


def safe_batches(rows, size):
    """Prevent duplicate queries/docs and known cross-query false negatives.

    rows contain query_id, positive_id, negative_id, and all known positive_ids
    (all grades above zero, not only grade 3). Unjudged false negatives remain
    possible and are disclosed; they are not silently relabeled irrelevant.
    """
    batch = []
    for row in rows:
        docs = {row["positive_id"], row["negative_id"]}
        if len(docs) != 2 or row["negative_id"] in row["positive_ids"]:
            raise ValueError("Negative is duplicate or known relevant")
        conflict = any(row["query_id"] == other["query_id"] or
                       docs & {other["positive_id"], other["negative_id"]} or
                       docs & other["positive_ids"] or
                       {other["positive_id"], other["negative_id"]} & row["positive_ids"]
                       for other in batch)
        if conflict or len(batch) >= size:
            if len(batch) >= 2:
                yield batch
            batch = []
        batch.append(row)
    if len(batch) >= 2:
        yield batch
