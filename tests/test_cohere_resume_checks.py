"""Numerical consistency regression checks without files, cleanup or AWS calls."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("resume_worker", Path(__file__).resolve().parents[1] / "scripts/resume_experiment_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
POLICY = {"minimum_cosine": .999, "maximum_abs_difference": .01,
          "reason": "Repeated identical provider requests vary numerically"}


class CohereResumeChecks(unittest.TestCase):
    def test_saved_chunk_accepts_valid_aligned_range(self):
        record = {"start": 55680, "stop": 55776, "sha256": "a" * 64}
        self.assertEqual(worker.validate_saved_chunk_range(record, 55680, 240765), (55680, 55776))

    def test_saved_chunk_rejects_bad_ranges_and_hash(self):
        for start, stop, digest in [(55679, 55775, "a" * 64),
                                    (55681, 55777, "a" * 64),
                                    (55680, 55775, "a" * 64),
                                    (240768, 240765, "a" * 64),
                                    (55680, 55776, "not-a-checksum")]:
            with self.subTest(start=start, stop=stop, digest=digest):
                with self.assertRaises(ValueError):
                    worker.validate_saved_chunk_range({"start": start, "stop": stop, "sha256": digest}, 55680, 240765)

    def test_missing_out_of_order_batch_is_not_skipped(self):
        saved = {55680, 55872}
        self.assertEqual(worker.remaining_chunk_starts(55680, 55968, saved), [55776])

    def test_legacy_check_reproduces_failure_for_small_normalized_variation(self):
        old = np.eye(2, dtype=np.float32)
        new = np.array([[1., .009], [.009, 1.]], dtype=np.float32)
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        with self.assertRaises(ValueError):
            worker.cohere_replay_check(new, old)
        result = worker.cohere_replay_check(new, old, POLICY)
        self.assertTrue(result["row_alignment_verified"])

    def test_changed_model_vectors_are_rejected(self):
        old = np.eye(2, dtype=np.float32)
        new = np.array([[.6, .8], [.8, .6]], dtype=np.float32)
        with self.assertRaises(ValueError):
            worker.cohere_replay_check(new, old, POLICY)

    def test_row_reordering_is_rejected(self):
        old = np.eye(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            worker.cohere_replay_check(old[::-1], old, POLICY)

    def test_nonfinite_output_is_rejected(self):
        old = np.eye(2, dtype=np.float32)
        new = old.copy(); new[0, 0] = np.nan
        with self.assertRaises(ValueError):
            worker.cohere_replay_check(new, old, POLICY)

    def test_zero_tail_and_prefix_boundary_are_verified(self):
        values = np.zeros((194, 1024), dtype=np.float32)
        values[:193, 0] = 1
        self.assertEqual(worker.validated_prefix(values), 192)
        values[-1, 0] = 1
        self.assertEqual(worker.validated_prefix(values), 194)
        values[100] = 0
        with self.assertRaises(ValueError):
            worker.validated_prefix(values)


if __name__ == "__main__":
    unittest.main()
