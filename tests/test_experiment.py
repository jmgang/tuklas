import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.corpus import Corpus, MappedQuery
from pycon_model2vec_demo.experiment import final_run_preflight, judged_candidate_rankings


class ExperimentTests(unittest.TestCase):
    def test_judged_candidate_ranking_is_explicitly_pool_only(self):
        corpus = Corpus(("a", "b", "c"), ("A", "B", "C"), {}, {}, 0)
        queries = (MappedQuery("q", "text", "geohazard", {"a": 1, "c": 0}),)
        vectors = np.asarray([[1, 0], [2, 0], [0, 1]], dtype=np.float32)
        ranked = judged_candidate_rankings(queries, np.asarray([[1, 0]], dtype=np.float32), vectors, corpus)
        self.assertEqual([row.corpus_id for row in ranked[0]], ["a", "c"])

    def test_final_preflight_refuses_without_five_local_models(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((FileNotFoundError, ValueError)):
                final_run_preflight(Path(directory))


if __name__ == "__main__":
    unittest.main()
