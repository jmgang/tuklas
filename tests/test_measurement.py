import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.measurement import (
    artifact_size_bytes, measure_warm_workload, peak_rss_bytes, repeated_timing,
)


class FakeEncoder:
    dimension = 2
    max_length = 16
    batch_size = 2

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_passages(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def count_truncated_passages(self, texts):
        return 0


class MeasurementTests(unittest.TestCase):
    def test_artifact_size_and_peak_rss(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a").write_bytes(b"1234")
            self.assertEqual(artifact_size_bytes(Path(directory)), 4)
        self.assertGreater(peak_rss_bytes(), 0)

    def test_repeated_warm_workload_reports_same_query_count(self):
        measurements = measure_warm_workload(
            FakeEncoder(), query_texts=["flood", "storm"], passage_texts=["a", "b"],
            passage_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            corpus_ids=["a", "b"], warmup=1, repetitions=2,
        )
        self.assertEqual(measurements["single_query_end_to_end"]["repetitions"], 4)
        self.assertEqual(measurements["passage_batch_encoding"]["texts_per_batch"], 2)
        self.assertGreaterEqual(measurements["single_query_exact_search"]["median_ms"], 0)

    def test_repeated_timing_validates_repetitions(self):
        self.assertEqual(repeated_timing(lambda: None, warmup=1, repetitions=3)["repetitions"], 3)
        with self.assertRaises(ValueError):
            repeated_timing(lambda: None, repetitions=0)


if __name__ == "__main__":
    unittest.main()
