import numpy as np

from pycon_model2vec_demo.full_rank_pca import (
    explained_variance_fraction,
    fit_full_rank_pca,
    transform_full_rank_pca,
    transform_pca_projection_batched,
)


def test_reduced_projection_matches_full_projection_prefix():
    values = np.random.default_rng(42).normal(size=(64, 8)).astype(np.float32)
    transform = fit_full_rank_pca(values, device="cpu", batch_size=13)
    expected = transform_full_rank_pca(values, transform)[:, :3]
    actual = transform_pca_projection_batched(
        values, transform, output_dimension=3, device="cpu", batch_size=11
    )
    assert np.allclose(actual, expected, atol=1e-5)
    retained = explained_variance_fraction(transform, 3)
    assert 0.0 < retained < 1.0
