"""Protect the raw-versus-mapped grade-3 selection rule."""

import unittest

from pycon_model2vec_demo.corpus import Corpus, MappedQuery
from pycon_model2vec_demo.data import Candidate, QueryGroup
from pycon_model2vec_demo.grade3 import score_query, select_cohorts
from pycon_model2vec_demo.retrieval import RankedPassage


class Grade3Tests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(("p1", "p2"), ("first", "second"), {"first": "p1", "second": "p2"}, {"first": "p1", "second": "p2"}, 0)

    def test_unmapped_second_gold_does_not_become_single_gold(self):
        groups = (
            QueryGroup("q1", "question one", (Candidate("a", "first", 3), Candidate("b", "missing", 3)), "geohazard"),
            QueryGroup("q2", "question two", (Candidate("c", "second", 3), Candidate("d", "first", 1)), "geohazard"),
        )
        mapped = (
            MappedQuery("q1", "question one", "geohazard", {"p1": 3}),
            MappedQuery("q2", "question two", "geohazard", {"p2": 3, "p1": 1}),
        )
        single, broad, audit = select_cohorts(groups, mapped, self.corpus)
        self.assertEqual([query.query_id for query in single], ["q2"])
        self.assertEqual([query.query_id for query in broad], ["q1", "q2"])
        self.assertEqual(audit["strict_selection_counts"]["multiple_raw_grade3"], 1)

    def test_one_raw_gold_that_does_not_map_is_excluded(self):
        groups = (QueryGroup("q1", "question one", (Candidate("a", "missing", 3),)), QueryGroup("q2", "question two", (Candidate("b", "first", 3),)))
        mapped = (MappedQuery("q1", "question one", None, {}), MappedQuery("q2", "question two", None, {"p1": 3}))
        single, broad, audit = select_cohorts(groups, mapped, self.corpus)
        self.assertEqual(len(single), 1)
        self.assertEqual(len(broad), 1)
        self.assertEqual(audit["strict_selection_counts"]["single_raw_grade3_unmapped"], 1)

    def test_single_gold_recall_and_unjudged(self):
        groups = (QueryGroup("q1", "question one", (Candidate("a", "first", 3), Candidate("b", "second", 1))),)
        mapped = (MappedQuery("q1", "question one", None, {"p1": 3, "p2": 1}),)
        single, _, _ = select_cohorts(groups, mapped, self.corpus)
        ranking = [RankedPassage(f"unknown-{index}", 1.0) for index in range(100)]
        ranking[1] = RankedPassage("p1", 0.9)
        ranking[2] = RankedPassage("p2", 0.8)
        metrics = score_query(single[0], ranking)
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        self.assertEqual(metrics["unjudged_at_10"], 0.8)


if __name__ == "__main__":
    unittest.main()
