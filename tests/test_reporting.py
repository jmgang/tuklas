import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.corpus import Corpus, MappedQuery
from pycon_model2vec_demo.evaluation import score_rankings
from pycon_model2vec_demo.reporting import build_report, illustrative_decisions, render_markdown
from pycon_model2vec_demo.retrieval import RankedPassage


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(("p1", "p2", "p3"), ("Relevant one", "Relevant two", "Unknown"), {}, {}, 0)
        self.queries = (
            MappedQuery("q1", "first question", "geohazard", {"p1": 2, "p2": 0}),
            MappedQuery("q2", "second question", "biological", {"p2": 1}),
        )
        teacher = ((RankedPassage("p1", 0.9), RankedPassage("p3", 0.8)), (RankedPassage("p2", 0.9),))
        student = ((RankedPassage("p3", 0.9), RankedPassage("p1", 0.8)), (RankedPassage("p2", 0.9),))
        self.rankings = {"teacher": teacher, "custom_student": student}
        self.scores = {name: score_rankings(self.queries, rows) for name, rows in self.rankings.items()}

    def test_report_keeps_unjudged_unknown_and_shows_disagreement(self):
        report = build_report(
            split="development", queries=self.queries, corpus=self.corpus, scores=self.scores,
            rankings=self.rankings, coverage={"unmatched": 2}, provenance={"dataset_revision": "test"},
            bootstrap_repetitions=100,
        )
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["mapped_judgment_coverage"]["unmatched"], 2)
        example = report["disagreement_examples"]["comparison_worse"][0]
        self.assertIsNone(example["retrieved"]["custom_student"][0]["relevance_if_judged"])
        self.assertIn("Unjudged@10", render_markdown(report))
        self.assertIn("Category breakdown", render_markdown(report))

    def test_decision_reports_all_failed_checks(self):
        measurements = {
            "teacher": {"warm": {"single_query_end_to_end": {"p95_ms": 100.0}}},
            "custom_student": {
                "warm": {"single_query_end_to_end": {"p95_ms": 20.0}},
                "model_artifact_size_bytes": 20 * 2**20,
                "peak_rss_after_workload_bytes": 2 * 2**30,
            },
        }
        thresholds = {
            "teacher_reference": "teacher",
            "quality": {"maximum_absolute_ndcg_at_10_loss": 0.01, "maximum_absolute_known_positive_recall_at_100_loss": 0.05},
            "operational": {
                "maximum_single_query_end_to_end_p95_ms": 100.0,
                "maximum_model_artifact_mib": 64.0,
                "maximum_peak_process_rss_gib": 4.0,
                "minimum_end_to_end_p95_speedup_over_teacher": 2.0,
            },
        }
        decision = illustrative_decisions(self.scores, measurements, thresholds)["custom_student"]
        self.assertFalse(decision["replacement_candidate"])
        self.assertFalse(decision["passes"]["ndcg_at_10_loss"])
        self.assertTrue(decision["passes"]["end_to_end_p95_speedup_over_teacher"])

    def test_rejects_misaligned_query_order(self):
        reversed_scores = dict(self.scores)
        reversed_scores["custom_student"] = dict(reversed_scores["custom_student"])
        reversed_scores["custom_student"]["per_query"] = list(reversed(reversed_scores["custom_student"]["per_query"]))
        with self.assertRaises(ValueError):
            build_report(split="development", queries=self.queries, corpus=self.corpus,
                         scores=reversed_scores, rankings=self.rankings, coverage={}, provenance={})


if __name__ == "__main__":
    unittest.main()
