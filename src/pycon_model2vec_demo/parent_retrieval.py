"""Exact segment retrieval, with one ranked result per original passage."""
import numpy as np
from .retrieval import RankedPassage, _best_indices


def shared_slices(text, policies):
    """One lossless layout validated against every model's passage budget."""
    if not text:
        raise ValueError('Empty source passage')
    slices=[]
    start=0
    while start<len(text):
        rest=text[start:]
        if all(p.fits(rest) for p in policies):
            end=len(text)
        else:
            # Each tokenizer supplies a safe candidate. Recheck the shortest
            # candidate against all tokenizers (token counts are not monotone).
            candidate=min(p.plan(rest)[0][1] for p in policies)
            while candidate>0 and not all(p.fits(rest[:candidate]) for p in policies):
                candidate-=1
            if not candidate:
                raise ValueError('No source character fits all frozen budgets')
            end=start+candidate
        slices.append([start,end])
        start=end
    if ''.join(text[a:b] for a,b in slices)!=text:
        raise AssertionError('Shared layout lost or duplicated original text')
    return slices


def parent_top_k(query_vectors,segment_vectors,owners,parent_ids,k=10,block_size=4096):
    """Score all segments and rank parents by their maximum segment score.

    Owners must be sorted original-passage row indices. Block-boundary splits
    are reduced correctly. Ten means ten unique parents, not ten raw segments.
    """
    q=np.asarray(query_vectors,dtype=np.float32)
    owners=np.asarray(owners,dtype=np.int64)
    if q.ndim!=2 or segment_vectors.ndim!=2 or q.shape[1]!=segment_vectors.shape[1]:
        raise ValueError('Embedding dimensions differ')
    if len(owners)!=len(segment_vectors) or not len(owners) or k<=0 or block_size<=0:
        raise ValueError('Invalid segment ownership or retrieval budget')
    if owners[0]<0 or owners[-1]>=len(parent_ids) or np.any(owners[1:]<owners[:-1]):
        raise ValueError('Owners must be sorted valid parent indices')
    if len(np.unique(owners))!=len(parent_ids) or len(set(parent_ids))!=len(parent_ids):
        raise ValueError('Each unique parent must have at least one segment')
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite query vectors')
    scores=np.full((len(q),len(parent_ids)),-np.inf,dtype=np.float32)
    for a in range(0,len(owners),block_size):
        b=min(a+block_size,len(owners))
        vectors=np.asarray(segment_vectors[a:b],dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError('Nonfinite segment vectors')
        similarity=q@vectors.T
        block_owners=owners[a:b]
        starts=np.r_[0,np.flatnonzero(block_owners[1:]!=block_owners[:-1])+1]
        reduced=np.maximum.reduceat(similarity,starts,axis=1)
        parents=block_owners[starts]
        scores[:,parents]=np.maximum(scores[:,parents],reduced)
    return tuple(tuple(RankedPassage(parent_ids[i],float(row[i]))
        for i in _best_indices(row,parent_ids,min(k,len(parent_ids)))) for row in scores)
