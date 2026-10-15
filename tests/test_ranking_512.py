"""All-in-memory safeguards for improved negative selection and batching."""
import unittest
import numpy as np
from pycon_model2vec_demo.ranking_512 import packed_safe_batches, hardest_judged_negatives


def row(query, positive, negative, known=None):
    return dict(query_id=query, positive_id=positive, negative_id=negative,
                positive_ids=set(known or [positive]))


class Ranking512Tests(unittest.TestCase):
    def test_conflicting_row_does_not_drop_other_queries(self):
        rows = [row('a', 'p1', 'n1'), row('b', 'p1', 'n2'), row('c', 'p3', 'n3')]
        batches = list(packed_safe_batches(rows, 2))
        self.assertEqual(sorted(r['query_id'] for b in batches for r in b), ['a', 'b', 'c'])
        self.assertEqual([len(b) for b in batches], [2, 1])

    def test_known_cross_query_positive_never_used_as_negative(self):
        rows = [row('a', 'p1', 'n1', ['p1', 'n2']), row('b', 'p2', 'n2')]
        self.assertEqual([len(b) for b in packed_safe_batches(rows, 4)], [1, 1])

    def test_duplicate_or_known_relevant_negative_rejected(self):
        with self.assertRaises(ValueError):
            list(packed_safe_batches([row('a', 'p1', 'p1')], 4))
        with self.assertRaises(ValueError):
            list(packed_safe_batches([row('a', 'p1', 'p2', ['p1', 'p2'])], 4))

    def test_hard_negative_only_uses_judged_candidates(self):
        candidates = ['judged-a', 'judged-b', 'judged-c']
        embeddings = np.array([[0., 1.], [1., 0.], [.8, .6]], dtype=np.float32)
        result = hardest_judged_negatives(np.array([1., 0.]), candidates, embeddings, 2)
        self.assertEqual(result, ['judged-b', 'judged-c'])

    def test_bad_negative_pool_rejected(self):
        with self.assertRaises(ValueError):
            hardest_judged_negatives(np.array([1., 0.]), ['a'], np.ones((2, 2)), 2)


if __name__ == '__main__':
    unittest.main()
