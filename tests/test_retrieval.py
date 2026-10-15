import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.corpus import Corpus
from pycon_model2vec_demo.retrieval import encode_corpus, exact_top_k, open_corpus_embeddings


class FakeEncoder:
    dimension = 2
    batch_size = 2

    def encode_passages(self, texts):
        return np.asarray([{"a": [1, 0], "b": [0, 1], "c": [1, 0]}[text] for text in texts], dtype=np.float32)

    def count_truncated_passages(self, texts):
        return 0


class RetrievalTests(unittest.TestCase):
    def test_full_collection_exact_top_k_with_stable_boundary_ties(self):
        ids = ("corpus_9", "corpus_2", "corpus_1", "corpus_3")
        vectors = np.asarray([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
        results = exact_top_k(np.asarray([[1, 0]], dtype=np.float32), vectors, ids, k=2, corpus_block_size=2)
        self.assertEqual([row.corpus_id for row in results[0]], ["corpus_1", "corpus_2"])

    def test_cache_is_model_specific_and_refuses_overwrite(self):
        corpus = Corpus(("corpus_0", "corpus_1", "corpus_2"), ("a", "b", "c"), {}, {}, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npy"
            encode_corpus(corpus, FakeEncoder(), path, corpus_sha256="corpus-sha", model_fingerprint="model-a")
            vectors = open_corpus_embeddings(path, corpus=corpus, corpus_sha256="corpus-sha", model_fingerprint="model-a")
            self.assertEqual(vectors.shape, (3, 2))
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["corpus_passage_count"], 3)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                open_corpus_embeddings(path, corpus=corpus, corpus_sha256="corpus-sha", model_fingerprint="model-b")
            with self.assertRaises(FileExistsError):
                encode_corpus(corpus, FakeEncoder(), path, corpus_sha256="corpus-sha", model_fingerprint="model-a")

    def test_exact_path_handles_full_released_corpus_size(self):
        count = 239704
        ids = tuple(f"corpus_{index}" for index in range(count))
        vectors = np.zeros((count, 2), dtype=np.float32)
        vectors[-1] = [1, 0]
        ranked = exact_top_k(np.asarray([[1, 0]], dtype=np.float32), vectors, ids, k=10)
        self.assertEqual(ranked[0][0].corpus_id, "corpus_239703")
        self.assertEqual(len(ranked[0]), 10)


if __name__ == "__main__":
    unittest.main()
