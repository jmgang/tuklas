"""Full-rank PCA helpers for the repaired SEA-LION static student.

Model2Vec applies PCA before SIF weighting.  Keeping all 1,024 dimensions does
not mean skipping PCA: centering is still material and the full component
matrix is an orthogonal change of basis.  These functions are batch-oriented
so a one-gigabyte vocabulary matrix can be fitted on a GPU without creating
another full in-memory copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


class MatrixLike(Protocol):
    shape: tuple[int, int]

    def __getitem__(self, key: slice) -> np.ndarray: ...


@dataclass(frozen=True)
class FullRankPCA:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    sample_count: int

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])


def _matrix_shape(matrix: MatrixLike | np.ndarray) -> tuple[int, int]:
    if len(matrix.shape) != 2:
        raise ValueError("PCA input must be a two-dimensional matrix")
    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    if rows < 2 or columns < 1:
        raise ValueError("PCA input must contain at least two rows and one column")
    return rows, columns


def _batch_tensor(matrix: MatrixLike | np.ndarray, start: int, stop: int, device: torch.device) -> torch.Tensor:
    values = np.asarray(matrix[start:stop], dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("PCA input contains a malformed or non-finite batch")
    return torch.from_numpy(np.ascontiguousarray(values)).to(device=device)


def fit_full_rank_pca(
    matrix: MatrixLike | np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 4096,
) -> FullRankPCA:
    """Fit centered, full-rank PCA using batchwise covariance accumulation.

    The output retains every input dimension.  No dimensional reduction is
    performed.  Eigenvectors receive a deterministic sign convention so the
    saved transform has a stable checksum across equivalent runs.
    """

    rows, columns = _matrix_shape(matrix)
    if batch_size <= 0:
        raise ValueError("PCA batch size must be positive")
    target = torch.device(device)

    total = torch.zeros(columns, dtype=torch.float64, device=target)
    for start in range(0, rows, batch_size):
        batch = _batch_tensor(matrix, start, min(start + batch_size, rows), target)
        total += batch.to(torch.float64).sum(dim=0)
    mean64 = total / rows

    # Accumulating the small d x d covariance in float64 avoids losing the
    # lower-variance directions while keeping the large n x d source float32.
    covariance = torch.zeros((columns, columns), dtype=torch.float64, device=target)
    for start in range(0, rows, batch_size):
        batch = _batch_tensor(matrix, start, min(start + batch_size, rows), target).to(torch.float64)
        centered = batch - mean64
        covariance.addmm_(centered.T, centered)
    covariance /= rows - 1

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    # sklearn's PCA stores components as rows.  eigh returns eigenvectors in
    # columns, so transpose after sorting.
    components = eigenvectors[:, order].T

    # Deterministic sign: the largest-magnitude loading in every component is
    # positive.  PCA signs are otherwise arbitrary.
    pivot_columns = torch.argmax(torch.abs(components), dim=1)
    pivot_values = components[torch.arange(columns, device=target), pivot_columns]
    signs = torch.where(pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values))
    components = components * signs[:, None]

    result = FullRankPCA(
        mean=mean64.to(torch.float32).cpu().numpy(),
        components=components.to(torch.float32).cpu().numpy(),
        explained_variance=eigenvalues.clamp_min(0).to(torch.float32).cpu().numpy(),
        sample_count=rows,
    )
    validate_full_rank_pca(result)
    return result


def validate_full_rank_pca(transform: FullRankPCA, *, atol: float = 2e-4) -> None:
    dimension = transform.dimension
    if transform.components.shape != (dimension, dimension):
        raise ValueError("full-rank PCA component matrix has the wrong shape")
    if transform.explained_variance.shape != (dimension,):
        raise ValueError("full-rank PCA variance vector has the wrong shape")
    if transform.sample_count < 2:
        raise ValueError("full-rank PCA sample count is invalid")
    if not all(np.isfinite(value).all() for value in (transform.mean, transform.components, transform.explained_variance)):
        raise ValueError("full-rank PCA transform contains non-finite values")
    identity = transform.components @ transform.components.T
    if not np.allclose(identity, np.eye(dimension, dtype=np.float32), atol=atol, rtol=atol):
        raise ValueError("full-rank PCA components are not orthonormal")


def transform_full_rank_pca(values: np.ndarray, transform: FullRankPCA) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != transform.dimension:
        raise ValueError("PCA transform input has the wrong shape")
    if not np.isfinite(values).all():
        raise ValueError("PCA transform input contains non-finite values")
    return np.ascontiguousarray((values - transform.mean) @ transform.components.T, dtype=np.float32)


def transform_full_rank_pca_batched(
    values: MatrixLike | np.ndarray,
    transform: FullRankPCA,
    *,
    device: str | torch.device,
    batch_size: int = 4096,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a full-rank transform in bounded GPU/CPU batches.

    A 170k by 1,024 float32 vocabulary is roughly 700 MB.  Transforming it in
    one NumPy expression creates another large temporary array.  This routine
    keeps the destination explicit and bounds device memory without changing
    the PCA mathematics.
    """

    rows, columns = _matrix_shape(values)
    validate_full_rank_pca(transform)
    if columns != transform.dimension:
        raise ValueError("PCA transform input has the wrong width")
    if batch_size <= 0:
        raise ValueError("PCA batch size must be positive")
    if output is None:
        output = np.empty((rows, columns), dtype=np.float32)
    if output.shape != (rows, columns) or output.dtype != np.float32:
        raise ValueError("PCA destination has the wrong shape or dtype")
    target = torch.device(device)
    mean = torch.from_numpy(np.ascontiguousarray(transform.mean)).to(target)
    components = torch.from_numpy(np.ascontiguousarray(transform.components)).to(target)
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            batch = _batch_tensor(values, start, stop, target)
            transformed = (batch - mean) @ components.T
            output[start:stop] = transformed.cpu().numpy()
    return output


def transform_pca_projection_batched(
    values: MatrixLike | np.ndarray,
    transform: FullRankPCA,
    *,
    output_dimension: int,
    device: str | torch.device,
    batch_size: int = 4096,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the leading PCA components without materializing the full projection."""

    rows, columns = _matrix_shape(values)
    validate_full_rank_pca(transform)
    if columns != transform.dimension:
        raise ValueError("PCA transform input has the wrong width")
    if not 0 < output_dimension <= columns:
        raise ValueError("PCA output dimension must be within the source width")
    if output is None:
        output = np.empty((rows, output_dimension), dtype=np.float32)
    if output.shape != (rows, output_dimension) or output.dtype != np.float32:
        raise ValueError("PCA projection destination has the wrong shape or dtype")
    target = torch.device(device)
    mean = torch.from_numpy(np.ascontiguousarray(transform.mean)).to(target)
    components = torch.from_numpy(
        np.ascontiguousarray(transform.components[:output_dimension])
    ).to(target)
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            batch = _batch_tensor(values, start, stop, target)
            output[start:stop] = ((batch - mean) @ components.T).cpu().numpy()
    return output


def explained_variance_fraction(transform: FullRankPCA, output_dimension: int) -> float:
    """Return the fraction of variance retained by the leading PCA dimensions."""

    if not 0 < output_dimension <= transform.dimension:
        raise ValueError("PCA output dimension must be within the source width")
    total = float(transform.explained_variance.sum())
    if total <= 0:
        raise ValueError("PCA explained variance is zero")
    return float(transform.explained_variance[:output_dimension].sum() / total)


def model2vec_sif_weights(vocabulary_size: int, sif_coefficient: float | None) -> np.ndarray:
    """Reproduce Model2Vec 0.9.0's deterministic Zipf/SIF weights."""

    if vocabulary_size <= 0:
        raise ValueError("vocabulary size must be positive")
    if sif_coefficient is None:
        return np.ones(vocabulary_size, dtype=np.float32)
    if sif_coefficient <= 0:
        raise ValueError("SIF coefficient must be positive")
    inverse_rank = 1.0 / np.arange(2, vocabulary_size + 2, dtype=np.float64)
    probabilities = inverse_rank / inverse_rank.sum()
    return np.asarray(sif_coefficient / (sif_coefficient + probabilities), dtype=np.float32)


def recover_pre_pca_embeddings(weighted_embeddings: np.ndarray, sif_coefficient: float | None) -> np.ndarray:
    """Undo only Model2Vec's deterministic SIF scaling.

    This is valid solely for an initialization whose provenance confirms PCA
    was disabled and vocabulary quantization was not used.
    """

    values = np.asarray(weighted_embeddings, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("weighted embedding matrix is invalid")
    weights = model2vec_sif_weights(values.shape[0], sif_coefficient)
    return np.ascontiguousarray(values / weights[:, None], dtype=np.float32)


def apply_pca_then_sif(
    raw_embeddings: np.ndarray,
    transform: FullRankPCA,
    sif_coefficient: float | None,
) -> np.ndarray:
    transformed = transform_full_rank_pca(raw_embeddings, transform)
    weights = model2vec_sif_weights(transformed.shape[0], sif_coefficient)
    return np.ascontiguousarray(transformed * weights[:, None], dtype=np.float32)
