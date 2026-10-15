"""In-memory checks only; no TemporaryDirectory or file deletion."""
import unittest
import torch
from tokenlearn.pretrain import StaticModelFineTuner
from pycon_model2vec_demo.tokenlearn_checkpoint import (
    checkpoint_safe_class, independent_tokenlearn_checkpoints,
)


class CheckpointTests(unittest.TestCase):
    def test_real_tokenlearn_snapshot_is_independent(self):
        cls = checkpoint_safe_class(StaticModelFineTuner)
        model = cls(torch.ones(3, 512), 512, 0)
        saved = model.state_dict()
        before = saved['w'].clone()
        with torch.no_grad():
            model.w.add_(1)
        self.assertTrue(torch.equal(saved['w'], before))
        model.load_state_dict(saved)
        self.assertTrue(torch.equal(model.w, before))

    def test_forward_and_export_path_unchanged(self):
        cls = checkpoint_safe_class(StaticModelFineTuner)
        torch.manual_seed(1)
        original = StaticModelFineTuner(torch.ones(3, 512), 512, 0)
        patched = cls(torch.ones(3, 512), 512, 0)
        patched.load_state_dict(original.state_dict())
        tokens = torch.tensor([[1, 2]])
        for left, right in zip(original(tokens), patched(tokens)):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(original.sub_forward(tokens), patched.sub_forward(tokens)))

    def test_scope_restored_after_failure(self):
        import tokenlearn.pretrain as pretrain
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with independent_tokenlearn_checkpoints():
                self.assertIsNot(pretrain.StaticModelFineTuner, StaticModelFineTuner)
                raise RuntimeError('intentional')
        self.assertIs(pretrain.StaticModelFineTuner, StaticModelFineTuner)
