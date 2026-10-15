"""Full-dimension objective and immediate bilingual regression guards."""

import unittest

from pycon_model2vec_demo import multiwikiqa_ranking_v3 as previous
from pycon_model2vec_demo import multiwikiqa_ranking_v4 as candidate


def row(en_recall, en_ndcg, tl_recall, tl_ndcg):
    return {'by_language': {'en': {'Recall@10': en_recall, 'nDCG@10': en_ndcg},
                            'tl': {'Recall@10': tl_recall, 'nDCG@10': tl_ndcg}}}


class ObjectiveProbeTests(unittest.TestCase):
    def test_only_full_dimension_allowed(self):
        candidate.validate_recipe({'learning_rate': .0005, 'epochs': 3,
                                   'matryoshka_dimensions': [1024]})
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            candidate.validate_recipe({'learning_rate': .0005, 'epochs': 3,
                                       'matryoshka_dimensions': [32, 64, 128, 256, 512, 1024]})

    def test_original_recipe_still_separate(self):
        previous.validate_recipe({'learning_rate': .0005, 'epochs': 3,
                                  'matryoshka_dimensions': [32, 64, 128, 256, 512, 1024]})

    def test_stops_either_language_metric_loss(self):
        base = row(.8, .7, .7, .6)
        self.assertTrue(candidate.has_development_regression(base, row(.8, .71, .69, .61)))
        self.assertTrue(candidate.has_development_regression(base, row(.8, .69, .71, .61)))
        self.assertFalse(candidate.has_development_regression(base, row(.8, .7, .7, .6)))
        self.assertFalse(candidate.has_development_regression(base, row(.81, .71, .71, .61)))

    def test_training_pairs_and_batches_unchanged(self):
        self.assertEqual(candidate.prepare.__code__.co_code, previous.prepare.__code__.co_code)
        self.assertEqual(candidate.batches.__code__.co_code, previous.batches.__code__.co_code)


if __name__ == '__main__':
    unittest.main()
