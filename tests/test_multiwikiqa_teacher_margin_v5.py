"""Development-only margin objective and frozen mining-manifest checks."""

import json
from pathlib import Path
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.sentence_transformer.losses import MarginMSELoss
from sentence_transformers.util import pairwise_cos_sim

from pycon_model2vec_demo import multiwikiqa_ranking_v4 as previous
from pycon_model2vec_demo import multiwikiqa_ranking_v5 as candidate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (ROOT / 'runs' / 'pycon-multiwikiqa-ranked-v3-20261004-165739-f68b9a'
            / 'teacher-guided-train-negatives-v2.json')


class TeacherMarginProbeTests(unittest.TestCase):
    def test_batching_and_development_guard_unchanged(self):
        self.assertEqual(candidate.batches.__code__.co_code, previous.batches.__code__.co_code)
        self.assertEqual(candidate.prepare.__code__.co_code, previous.prepare.__code__.co_code)
        self.assertEqual(candidate.has_development_regression.__code__.co_code,
                         previous.has_development_regression.__code__.co_code)

    def test_mining_manifest_is_training_only(self):
        report = json.loads(MANIFEST.read_text())
        rows = report['rows']
        self.assertEqual((report['train_pairs_available'], report['triplets'], len(rows)),
                         (1734, 1726, 1726))
        self.assertEqual(len({row['query_id'] for row in rows}), len(rows))
        self.assertTrue(all(row['teacher_margin'] >= .05 for row in rows))
        self.assertEqual(len({row['negative_parent_id'] for row in rows}), 201)
        self.assertEqual(report['negative_source_restriction'],
                         'only development-training source articles; validation and heldout articles excluded')

    def test_official_margin_loss_has_finite_backward(self):
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, 'query': 1, 'yes': 2, 'no': 3}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        torch.manual_seed(42)
        model = SentenceTransformer(
            modules=[StaticEmbedding(tokenizer, embedding_weights=torch.randn(4, 1024))],
            device='cpu')
        features = [model.tokenize(texts) for texts in
                    (['query', 'query'], ['yes', 'yes'], ['no', 'no'])]
        loss = MarginMSELoss(model, similarity_fct=pairwise_cos_sim)(
            features, torch.tensor([.1, .2], dtype=torch.float32))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
