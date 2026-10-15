"""Distinguish learned vector changes from official fixed-weight export scaling."""
import numpy as np


def compare_block(initial, student, row_start, pad_id):
    if initial.shape != student.shape or initial.dtype != np.float32 or student.dtype != np.float32:
        raise ValueError("Weight audit shape/precision mismatch")
    if not np.isfinite(initial).all() or not np.isfinite(student).all():
        raise ValueError("Nonfinite weights")
    # Pinned official recipe freezes zero-initialized token weights: sigmoid(0)=.5.
    # The padding weight is -10000, so its fixed export multiplier is zero.
    expected = initial * np.float32(0.5)
    if row_start <= pad_id < row_start + len(initial):
        expected[pad_id-row_start] = 0
    nonpad = np.arange(row_start,row_start+len(initial)) != pad_id
    delta = np.any(student != expected,axis=1) & nonpad
    a_norm = np.sum(initial*initial,axis=1,dtype=np.float64)
    b_norm = np.sum(student*student,axis=1,dtype=np.float64)
    valid = (a_norm > 0) & (b_norm > 0) & nonpad
    cosine = np.ones(len(initial),dtype=np.float64)
    cosine[valid] = (np.sum(initial[valid]*student[valid],axis=1,dtype=np.float64)
                     / np.sqrt(a_norm[valid]*b_norm[valid]))
    return {"rows":len(initial), "changed_beyond_fixed_export_scaling":int(delta.sum()),
            "direction_changed_cosine_below_0_99999":int(((cosine<0.99999)&valid).sum()),
            "zero_initial_became_nonzero":int(((a_norm==0)&(b_norm>0)&nonpad).sum())}
