import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.data import Candidate, QueryGroup, split_query_groups
from pycon_model2vec_demo.ranking import ndcg_at_k, rank_group


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.group = QueryGroup(
            "q1",
            "flood protection",
            (
                Candidate("p2", "irrelevant", 0),
                Candidate("p1", "floodproofing", 2),
            ),
        )

    def test_relevant_passage_ranks_first(self):
        encoder = FakeEncoder(
            {"flood protection": [1.0, 0.0], "irrelevant": [0.0, 1.0], "floodproofing": [1.0, 0.0]}
        )
        ranked = rank_group(self.group, encoder)
        self.assertEqual(ranked[0].candidate.passage_id, "p1")
        self.assertAlmostEqual(ndcg_at_k(ranked), 1.0)

    def test_ties_use_stable_passage_id(self):
        encoder = FakeEncoder({text: [1.0, 0.0] for text in ("flood protection", "irrelevant", "floodproofing")})
        self.assertEqual([item.candidate.passage_id for item in rank_group(self.group, encoder)], ["p1", "p2"])

    def test_no_positive_judgment_is_undefined(self):
        group = QueryGroup("q2", "query", (Candidate("p", "passage", 0),))
        ranked = rank_group(group, FakeEncoder({"query": [1.0], "passage": [1.0]}))
        self.assertIsNone(ndcg_at_k(ranked))

    def test_split_is_deterministic_and_disjoint(self):
        groups = [QueryGroup(str(i), f"query {i}", (Candidate(f"p{i}", "text", 1),)) for i in range(100)]
        first = split_query_groups(groups, seed="demo", evaluation_fraction=0.2)
        second = split_query_groups(groups, seed="demo", evaluation_fraction=0.2)
        self.assertEqual(first, second)
        self.assertFalse({g.query_id for g in first[0]} & {g.query_id for g in first[1]})
        self.assertTrue(first[0] and first[1])

    def test_duplicate_query_text_stays_in_one_split(self):
        groups = [
            QueryGroup("a", "Storm surge", (Candidate("p1", "text", 1),)),
            QueryGroup("b", " storm   SURGE ", (Candidate("p2", "text", 1),)),
        ]
        development, evaluation = split_query_groups(groups, seed="demo", evaluation_fraction=0.5)
        self.assertIn(len(development), (0, 2))
        self.assertIn(len(evaluation), (0, 2))


if __name__ == "__main__":
    unittest.main()
