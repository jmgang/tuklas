import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.encoders import E5Encoder, StaticEncoder


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.tile([0.6, 0.8], (len(texts), 1))


class FakeStaticModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.tile([0.0, 1.0], (len(texts), 1))


class EncoderTests(unittest.TestCase):
    def test_e5_roles_and_batch_settings(self):
        fake = FakeSentenceTransformer()
        encoder = E5Encoder(fake, dimension=2, batch_size=3)
        self.assertEqual(encoder.encode_queries(["storm", "flood"]).shape, (2, 2))
        self.assertEqual(encoder.encode_passages(["levee"]).shape, (1, 2))
        self.assertEqual(fake.calls[0][0], ["query: storm", "query: flood"])
        self.assertEqual(fake.calls[1][0], ["passage: levee"])
        self.assertEqual(fake.calls[0][1]["batch_size"], 3)
        self.assertTrue(fake.calls[0][1]["normalize_embeddings"])

    def test_static_roles_and_batch_settings(self):
        fake = FakeStaticModel()
        encoder = StaticEncoder(
            fake, dimension=2, batch_size=7, query_prefix="query: ", passage_prefix="passage: "
        )
        self.assertEqual(encoder.encode_queries(["storm"]).shape, (1, 2))
        self.assertEqual(encoder.encode_passages(["levee", "dam"]).shape, (2, 2))
        self.assertEqual(fake.calls[0][0], ["query: storm"])
        self.assertEqual(fake.calls[1][0], ["passage: levee", "passage: dam"])
        self.assertEqual(fake.calls[1][1]["batch_size"], 7)
        self.assertEqual(fake.calls[1][1]["max_length"], 512)
        self.assertNotIn("normalize", fake.calls[1][1])

    def test_potion_has_no_role_prefixes(self):
        fake = FakeStaticModel()
        encoder = StaticEncoder(fake, dimension=2)
        encoder.encode_queries(["storm"])
        encoder.encode_passages(["levee"])
        self.assertEqual([call[0] for call in fake.calls], [["storm"], ["levee"]])

    def test_empty_input_preserves_dimension_without_calling_model(self):
        for encoder in (E5Encoder(FakeSentenceTransformer(), dimension=2), StaticEncoder(FakeStaticModel(), dimension=2)):
            self.assertEqual(encoder.encode_queries([]).shape, (0, 2))
            self.assertEqual(encoder.encode_passages([]).shape, (0, 2))
            self.assertEqual(encoder.model.calls, [])

    def test_invalid_shape_or_normalization_is_rejected(self):
        fake = FakeSentenceTransformer()
        fake.encode = lambda texts, **kwargs: np.ones((len(texts), 3))
        with self.assertRaisesRegex(ValueError, "Expected"):
            E5Encoder(fake, dimension=2).encode_queries(["storm"])
        fake.encode = lambda texts, **kwargs: np.ones((len(texts), 2))
        with self.assertRaisesRegex(ValueError, "L2-normalized"):
            E5Encoder(fake, dimension=2).encode_queries(["storm"])


if __name__ == "__main__":
    unittest.main()
