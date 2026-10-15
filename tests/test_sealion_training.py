"""In-memory tests: no temporary directories or cleanup operations."""
TOKENLEARN_COMMIT = "e0c3d28bd46a7fe6db84259d10201f8c50596f97"
import unittest
from unittest.mock import patch
import runpy
import sys
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from pycon_model2vec_demo.sealion_training import (
    article_partition, safe_windows, pad_collate, official_training_adapter,
)



def _tokenlearn_git_pin_installed(expected_commit):
    """True only when tokenlearn is installed from the frozen Git commit.

    The SageMaker container installs tokenlearn from the pinned commit, so
    ``verify_runtime`` passes there and every completed job is evidence of it.
    A developer workstation typically has the PyPI release instead, which has
    no VCS provenance, so the pin check correctly refuses.  Tests that exercise
    the *real* pinned library are skipped in that case rather than loosened:
    a genuine pin mismatch in a provisioned environment must still fail loudly.
    """
    import importlib.metadata
    import json as _json
    try:
        direct_url = importlib.metadata.distribution("tokenlearn").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return False
    if not direct_url:
        return False
    return _json.loads(direct_url).get("vcs_info", {}).get("commit_id") == expected_commit

class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text))) + ([999, 1000] if add_special_tokens else [])


class SeaLionTrainingTests(unittest.TestCase):
    def test_split_preserves_unicode_and_offsets(self):
        text = "Pilipinas 🇵🇭. café. " * 20
        windows = safe_windows(text, CharacterTokenizer(), max_tokens=37, max_chars=99)
        self.assertEqual("".join(x[2] for x in windows), text)
        for start, end, fragment in windows:
            self.assertEqual(text[start:end], fragment)
            self.assertLessEqual(len(fragment) + 2, 37)

    def test_short_text_unchanged(self):
        self.assertEqual(safe_windows("English / Tagalog", CharacterTokenizer()),
                         [(0, 17, "English / Tagalog")])

    def test_empty_source_refused(self):
        with self.assertRaises(ValueError):
            safe_windows("  ", CharacterTokenizer())

    def test_article_split_stable_and_shared(self):
        self.assertEqual(article_partition("tl/same"), article_partition("tl/same"))
        assignments = [article_partition(str(i)) for i in range(1000)]
        self.assertTrue(70 < assignments.count("validation") < 130)

    def test_collate_uses_nonzero_pad(self):
        x, y = pad_collate([([2, 3], torch.ones(4)), ([2], torch.zeros(4))], 1)
        self.assertEqual(x.tolist(), [[2, 3], [2, 1]])
        self.assertEqual(tuple(y.shape), (2, 4))

    def make_model(self):
        from model2vec import StaticModel
        tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "one": 2, "two": 3}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        return StaticModel(vectors=np.ones((4, 4), dtype=np.float32), tokenizer=tokenizer, normalize=True)

    def test_actual_model_padding_and_collation_are_invariant(self):
        import model2vec.train as train
        import model2vec.train.dataset as dataset
        with official_training_adapter(["two"], np.ones((1, 4), dtype=np.float32), "<pad>"):
            m = train.StaticModelForSimilarity.from_static_model(model=self.make_model(), out_dim=4, n_layers=0)
            self.assertEqual(m.pad_id, 1)
            a = m._encode(torch.tensor([[2]]))
            b = m._encode(torch.tensor([[2, 1]]))
            self.assertTrue(torch.equal(a, b))
            padded, _ = dataset.TextDataset.collate_fn([([2, 3], torch.ones(4)), ([2], torch.ones(4))])
            self.assertEqual(padded[1, 1].item(), 1)

    def test_scoped_restoration_on_failure(self):
        import model2vec.train as train
        import model2vec.train.dataset as dataset
        old_class, old_collate = train.StaticModelForSimilarity, dataset.TextDataset.collate_fn
        with self.assertRaises(RuntimeError):
            with official_training_adapter([], np.ones((0, 4), dtype=np.float32), "<pad>"):
                raise RuntimeError("intentional")
        self.assertIs(train.StaticModelForSimilarity, old_class)
        self.assertIs(dataset.TextDataset.collate_fn, old_collate)

    def test_real_fit_runs_with_article_validation(self):
        import model2vec.train as train
        metrics = []
        with official_training_adapter(["one two"], np.ones((1, 4), dtype=np.float32), "<pad>", metrics.append):
            m = train.StaticModelForSimilarity.from_static_model(model=self.make_model(), out_dim=4, n_layers=0,
                                                                 freeze_weights=True)
            m.fit(["one", "two"], torch.ones((2, 4)), batch_size=2, device="cpu", max_epochs=1)
            self.assertTrue(metrics)
            self.assertTrue(torch.isfinite(torch.from_numpy(m.to_static_model().embedding)).all())

    def test_adapter_restores_lowest_validation_loss_checkpoint(self):
        import model2vec.train as train
        import model2vec.train.trainer as trainer

        validation_values = iter((0.10, 0.50))
        selected_weights = []

        def measured_validation(model, *args, **kwargs):
            value = next(validation_values)
            if value == 0.10:
                selected_weights.append({
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                })
            return {"val_loss": value}

        with patch.object(trainer, "_run_validation", measured_validation):
            with official_training_adapter(
                ["one two"], np.ones((1, 4), dtype=np.float32), "<pad>"
            ) as state:
                model = train.StaticModelForSimilarity.from_static_model(
                    model=self.make_model(), out_dim=4, n_layers=0, freeze_weights=True
                )
                model.fit(
                    ["one", "two"], torch.ones((2, 4)), batch_size=2,
                    device="cpu", max_epochs=2, early_stopping_patience=None,
                )
                self.assertEqual(state["selected_validation_index"], 0)
                self.assertAlmostEqual(state["best_validation_value"], 0.10)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(), selected_weights[0][key]))

    def test_adapter_can_select_checkpoint_by_development_retrieval(self):
        import model2vec.train as train
        import model2vec.train.trainer as trainer

        validation_values = iter((0.10, 0.50))
        retrieval_values = iter((0.40, 0.80))
        selected_weights = []

        def measured_validation(model, *args, **kwargs):
            value = next(validation_values)
            if value == 0.50:
                selected_weights.append({
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                })
            return {"val_loss": value}

        def retrieval_selector(model, event_index, metrics):
            return {
                "selection_value": next(retrieval_values),
                "selection_metric": "development_recall_at_10_plus_ndcg_at_10",
                "event_index": event_index,
                "val_loss": float(metrics["val_loss"]),
            }

        with patch.object(trainer, "_run_validation", measured_validation):
            with official_training_adapter(
                ["one two"], np.ones((1, 4), dtype=np.float32), "<pad>",
                checkpoint_selector=retrieval_selector,
            ) as state:
                model = train.StaticModelForSimilarity.from_static_model(
                    model=self.make_model(), out_dim=4, n_layers=0, freeze_weights=True
                )
                model.fit(
                    ["one", "two"], torch.ones((2, 4)), batch_size=2,
                    device="cpu", max_epochs=2, early_stopping_patience=None,
                )
                self.assertEqual(state["selection_mode"], "development_retrieval")
                self.assertEqual(state["selected_validation_index"], 1)
                self.assertAlmostEqual(state["best_selection_value"], 0.80)
                self.assertEqual(len(state["selection_events"]), 2)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(), selected_weights[0][key]))

    def test_nonfinite_initialization_refused(self):
        import model2vec.train as train
        model = self.make_model()
        model.embedding[2, 1] = np.nan
        with official_training_adapter([], np.ones((0, 4), dtype=np.float32), "<pad>"):
            with self.assertRaises(ValueError):
                train.StaticModelForSimilarity.from_static_model(model=model, out_dim=4)

    @unittest.skipUnless(
        _tokenlearn_git_pin_installed(TOKENLEARN_COMMIT),
        "tokenlearn is not installed from the frozen Git commit",
    )
    def test_pinned_tokenlearn_cli_calls_real_training_and_exports(self):
        from model2vec import StaticModel
        import model2vec.train as train
        import tokenlearn.utils as official_utils
        model = self.make_model()
        saved = []
        real_fit = train.StaticModelForSimilarity.fit
        def bounded_test_fit(instance, *args, **kwargs):
            kwargs["max_epochs"] = 1
            return real_fit(instance, *args, **kwargs)
        args = ["tokenlearn.train", "--model2vec-model-name", "memory-only",
                "--data-path", "memory-only", "--save-path", "memory-only-output",
                "--device", "cpu", "--batch-size", "2", "--vocab-size", "0"]
        with patch.object(official_utils, "collect_means_and_texts", return_value=(["one", "two"], np.ones((2,4), dtype=np.float32))), \
             patch.object(StaticModel, "from_pretrained", return_value=model), \
             patch.object(StaticModel, "save_pretrained", lambda instance, path: saved.append(instance.embedding.copy())), \
             patch.object(train.StaticModelForSimilarity, "fit", bounded_test_fit), \
             patch.object(sys, "argv", args):
            with official_training_adapter(["one two"], np.ones((1,4), dtype=np.float32), "<pad>"):
                runpy.run_module("tokenlearn.train", run_name="__main__")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].shape, (4,4))
        self.assertFalse(np.array_equal(saved[0], model.embedding))

    @unittest.skipUnless(
        _tokenlearn_git_pin_installed(TOKENLEARN_COMMIT),
        "tokenlearn is not installed from the frozen Git commit",
    )
    def test_real_tokenlearn_feature_target_is_token_mean(self):
        import tokenlearn.featurize as official
        class Teacher:
            def encode(self, texts, **kwargs):
                self.called = kwargs
                return [torch.tensor([[99.,99.], [2.,4.], [4.,8.], [99.,99.]]) for _ in texts]
        teacher = Teacher()
        with patch.object(official, "_truncate_text", lambda tokenizer, text: text):
            rows = list(official._encode(["original text"], teacher, object()))
        self.assertEqual(rows[0][0], "original text")
        self.assertTrue(np.array_equal(rows[0][1], np.array([3.,6.])))
        self.assertEqual(teacher.called["output_value"], "token_embeddings")
