import numpy as np

from pycon_model2vec_demo.full_rank_pca import (
    apply_pca_then_sif,
    fit_full_rank_pca,
    model2vec_sif_weights,
    recover_pre_pca_embeddings,
    transform_full_rank_pca,
    transform_full_rank_pca_batched,
)


def test_full_rank_pca_centers_without_reducing_dimensions():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(48, 7)).astype(np.float32)
    transform = fit_full_rank_pca(values, device="cpu", batch_size=11)
    transformed = transform_full_rank_pca(values, transform)

    assert transformed.shape == values.shape
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=2e-6)
    np.testing.assert_allclose(
        transformed.var(axis=0, ddof=1),
        transform.explained_variance,
        rtol=2e-4,
        atol=2e-5,
    )


def test_full_rank_pca_preserves_centered_pairwise_cosines():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(64, 9)).astype(np.float32)
    transform = fit_full_rank_pca(values, device="cpu", batch_size=13)
    centered = values - values.mean(axis=0)
    transformed = transform_full_rank_pca(values, transform)

    def cosine_matrix(matrix):
        normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        return normalized @ normalized.T

    np.testing.assert_allclose(cosine_matrix(centered), cosine_matrix(transformed), atol=3e-5, rtol=3e-5)


def test_batched_transform_matches_direct_transform():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(57, 8)).astype(np.float32)
    transform = fit_full_rank_pca(values, device="cpu", batch_size=9)
    expected = transform_full_rank_pca(values, transform)
    actual = transform_full_rank_pca_batched(values, transform, device="cpu", batch_size=7)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_sif_recovery_and_reapplication_round_trip_without_pca():
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(31, 5)).astype(np.float32)
    coefficient = 1e-4
    weights = model2vec_sif_weights(raw.shape[0], coefficient)
    stored = raw * weights[:, None]
    recovered = recover_pre_pca_embeddings(stored, coefficient)
    np.testing.assert_allclose(recovered, raw, atol=2e-6, rtol=2e-6)

    transform = fit_full_rank_pca(raw, device="cpu", batch_size=8)
    repaired = apply_pca_then_sif(recovered, transform, coefficient)
    expected = transform_full_rank_pca(raw, transform) * weights[:, None]
    np.testing.assert_allclose(repaired, expected, atol=2e-6, rtol=2e-6)
