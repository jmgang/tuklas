"""As-released bilingual retrieval evaluation; no training or relabeling."""
from collections import Counter
import math
import numpy as np

CUTOFFS = (1, 3, 5, 10)


def normalized(vectors):
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid embedding matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def rank_parents(query_vectors, segment_vectors, owners, parent_ids):
    """Exact cosine ranking; max segment per parent, deterministic ties."""
    q, d = normalized(query_vectors), normalized(segment_vectors)
    owners = np.asarray(owners, dtype=np.int64)
    if q.shape[1] != d.shape[1] or len(owners) != len(d):
        raise ValueError("Embedding/ownership mismatch")
    if (len(set(parent_ids)) != len(parent_ids) or not len(owners)
            or np.any(owners[1:] < owners[:-1]) or owners[0] != 0
            or owners[-1] != len(parent_ids)-1
            or len(np.unique(owners)) != len(parent_ids)):
        raise ValueError("Require sorted owners covering unique parents")
    similarities = q @ d.T
    starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1])+1]
    result = []
    names = np.asarray(parent_ids)
    for row in similarities:
        best = [int(a + np.argmax(row[a:b])) for a,b in
                zip(starts, np.r_[starts[1:],len(owners)])]
        scores = row[best]
        order = np.lexsort((names,-scores))[:min(10,len(parent_ids))]
        result.append([(parent_ids[i],best[i],float(scores[i])) for i in order])
    return result


def score(query, ranking, chunks):
    ids = [x[0] for x in ranking]
    if len(ids)>10 or len(ids)!=len(set(ids)):
        raise ValueError("Require at most ten unique source articles")
    gold = query["gold_parent_id"]
    rank = ids.index(gold)+1 if gold in ids else None
    metrics = {f"Recall@{k}":float(rank is not None and rank<=k) for k in CUTOFFS}
    metrics.update({"MRR@10":1/rank if rank else 0.0,
                    "nDCG@10":1/math.log2(rank+1) if rank else 0.0})
    spans = query["answer_spans"]
    covered = False
    for parent, index, _ in ranking:
        piece = chunks[index]
        if piece["parent_id"] != parent:
            raise ValueError("Returned chunk/parent mismatch")
        if parent == gold:
            covered = any(piece["start"]<=s["start"] and s["end"]<=piece["end"] for s in spans)
    metrics.update({f"AnswerSpanCoverage@{k}":
                    float(covered and rank is not None and rank<=k) if spans else None
                    for k in CUTOFFS})
    return metrics


def summarize(rows):
    groups = {"all_10000":rows,
              "structurally_eligible": [r for r in rows if not r["exclusion_reasons"]],
              "evaluation_partition": [r for r in rows if r["partition"]=="evaluation"],
              "development_partition": [r for r in rows if r["partition"]=="development"],
              "English": [r for r in rows if r["language"]=="en"],
              "Tagalog": [r for r in rows if r["language"]=="tl"]}
    out = {}
    for name,group in groups.items():
        if not group:
            continue
        metrics = {}
        for key in group[0]["metrics"]:
            values=[r["metrics"][key] for r in group if r["metrics"][key] is not None]
            metrics[key]={"mean":float(np.mean(values)) if values else None,"denominator":len(values)}
        out[name]={"questions":len(group),"source_articles":len({r["gold_parent_id"] for r in group}),
                   "metrics":metrics}
    return out


def paired_cluster_ci(left, right, metric, repetitions=2000, seed=42):
    """Paired article-cluster bootstrap of query-weighted mean differences."""
    if [r["query_id"] for r in left]!=[r["query_id"] for r in right]:
        raise ValueError("Different question ordering/cohort")
    ids=sorted({r["gold_parent_id"] for r in left})
    index={name:i for i,name in enumerate(ids)}
    totals=np.zeros(len(ids),dtype=np.float64)
    counts=np.zeros(len(ids),dtype=np.int64)
    for a,b in zip(left,right):
        if a["gold_parent_id"]!=b["gold_parent_id"]:
            raise ValueError("Gold article changed")
        if a["metrics"][metric] is None or b["metrics"][metric] is None:
            continue
        i=index[a["gold_parent_id"]]
        totals[i]+=a["metrics"][metric]-b["metrics"][metric]
        counts[i]+=1
    if not counts.sum():
        raise ValueError("No comparable observations")
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repetitions):
        draws=rng.integers(0,len(ids),size=len(ids))
        denom=counts[draws].sum()
        if denom:
            samples.append(totals[draws].sum()/denom)
    lo,hi=np.quantile(samples,[.025,.975])
    return {"delta":float(totals.sum()/counts.sum()),"ci95":[float(lo),float(hi)],
            "unit":"fraction, left minus right", "cluster":"source article",
            "seed":seed,"repetitions":repetitions,"questions":int(counts.sum()),
            "excludes_zero":bool(lo>0 or hi<0)}


def validate_questions(queries, parents):
    if len(queries)!=10000 or len(parents)!=1294:
        raise ValueError("Dataset counts changed")
    if Counter(q["language"] for q in queries)!={"en":4999,"tl":5001}:
        raise ValueError("Language cohort changed")
    ids={p["parent_id"] for p in parents}
    if len({q["query_id"] for q in queries})!=10000:
        raise ValueError("Duplicate query identities")
    for q in queries:
        if not q["question"].strip() or q["gold_parent_id"] not in ids:
            raise ValueError("Unscorable query/source identity")
        if set(q["exclusion_reasons"])-{"empty-answer-reference"}:
            raise ValueError("Unexpected structural defect requires review")
