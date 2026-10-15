import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.corpus import MappedQuery
from pycon_model2vec_demo.evaluation import (
    bootstrap_mean_interval, paired_bootstrap_difference, paired_quality_intervals,
    quality_intervals, score_rankings,
)
from pycon_model2vec_demo.retrieval import RankedPassage


class EvaluationTests(unittest.TestCase):
    def test_known_positive_and_unjudged_passage(self):
        query = MappedQuery("q1", "flood shelter", "geohazard", {"p1": 2, "p2": 0})
        ranked = ((RankedPassage("p1", 0.9), RankedPassage("p3", 0.8)),)
        report = score_rankings((query,), ranked)
        metrics = report["overall"]
        self.assertAlmostEqual(metrics["nDCG@10"], 1.0)
        self.assertAlmostEqual(metrics["R@10"], 1.0)
        self.assertAlmostEqual(metrics["RR@10"], 1.0)
        self.assertAlmostEqual(metrics["Hit@10"], 1.0)
        self.assertGreater(metrics["Unjudged@10"], 0.0)

    def test_paired_difference_preserves_query_identity(self):
        first = {"per_query": [{"query_id": "a", "metrics": {"nDCG@10": 0.2}}, {"query_id": "b", "metrics": {"nDCG@10": 0.8}}]}
        second = {"per_query": [{"query_id": "b", "metrics": {"nDCG@10": 0.9}}, {"query_id": "a", "metrics": {"nDCG@10": 0.4}}]}
        result = paired_bootstrap_difference(first, second, repetitions=100)
        self.assertAlmostEqual(result["mean_difference"], 0.15)

    def test_uncertainty_resamples_queries_and_splits_categories(self):
        queries = (
            MappedQuery("q1", "flood", "geohazard", {"p1": 2}),
            MappedQuery("q2", "storm", "meteorological_hydrological", {"p2": 1}),
        )
        good = score_rankings(queries, ((RankedPassage("p1", 1.0),), (RankedPassage("p2", 1.0),)))
        mixed = score_rankings(queries, ((RankedPassage("p1", 1.0),), (RankedPassage("x", 1.0),)))
        intervals = quality_intervals(good, repetitions=100)
        self.assertEqual(intervals["overall"]["nDCG@10"]["query_count"], 2)
        self.assertEqual(intervals["by_category"]["geohazard"]["nDCG@10"]["query_count"], 1)
        paired = paired_quality_intervals(good, mixed, repetitions=100)
        self.assertLess(paired["metrics"]["nDCG@10"]["mean_difference"], 0)
        interval = bootstrap_mean_interval(mixed, metric="nDCG@10", repetitions=100)
        self.assertAlmostEqual(interval["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
