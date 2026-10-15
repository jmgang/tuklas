"""In-memory checks only; no TemporaryDirectory cleanup or AWS."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pycon_model2vec_demo.retrieval_finetuning import independent_state, safe_batches, split_development


class RetrievalTrainingChecks(unittest.TestCase):
    def test_best_state_is_independent_of_later_training(self):
        import torch
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1)
        saved = independent_state(module)
        with torch.no_grad():
            module.weight.fill_(9)
        self.assertEqual(saved["weight"].item(), 1)

    def test_evaluation_query_cannot_enter_training(self):
        with self.assertRaisesRegex(ValueError, "Evaluation"):
            split_development([SimpleNamespace(query_id="test")], {"test"})

    def test_development_partitions_are_disjoint_and_repeatable(self):
        groups = [SimpleNamespace(query_id=str(i)) for i in range(100)]
        train, val = split_development(groups, {"unseen"})
        again = split_development(groups, {"unseen"})
        self.assertEqual((train, val), again)
        self.assertFalse({x.query_id for x in train} & {x.query_id for x in val})
        self.assertEqual(len(train) + len(val), 100)

    def test_known_relevant_documents_cannot_be_cross_query_negatives(self):
        rows = [{"query_id": str(i), "positive_id": f"p{i}", "negative_id": f"n{i}",
                 "positive_ids": {f"p{i}"}} for i in range(8)]
        rows[2]["positive_ids"].add("p1")
        batches = list(safe_batches(rows, 4))
        for batch in batches:
            for row in batch:
                for other in batch:
                    if row is not other:
                        self.assertFalse({other["positive_id"], other["negative_id"]} & row["positive_ids"])

    def test_known_positive_cannot_be_explicit_negative(self):
        bad = {"query_id": "q", "positive_id": "p", "negative_id": "n", "positive_ids": {"p", "n"}}
        with self.assertRaisesRegex(ValueError, "known relevant"):
            list(safe_batches([bad], 4))

    def test_real_training_module_and_model2vec_export_agree(self):
        import importlib.util
        import numpy as np
        from model2vec import StaticModel
        from tokenizers import Tokenizer, models, pre_tokenizers
        from sentence_transformers import SentenceTransformer
        path = Path(__file__).resolve().parents[1] / "scripts/run_local_retrieval_repair.py"
        spec = importlib.util.spec_from_file_location("retrieval_repair_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]":0,"query":1,"passage":2,":":3,
            "oysters":4,"acidification":5},unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        base = StaticModel(vectors=np.random.default_rng(7).normal(size=(6,32)).astype(np.float32),
                           tokenizer=tokenizer,normalize=True)
        module = runner.ExportParityEmbedding(base)
        model = SentenceTransformer(modules=[module],device="cpu")
        self.assertLess(runner.check_parity(model,base),2e-6)
        self.assertLess(runner.check_parity(model,runner.export(module,base)),2e-6)

    def test_segment_progress_event_has_no_directory_argument_collision(self):
        import importlib.util
        from unittest.mock import patch
        path = Path(__file__).resolve().parents[1] / "scripts/run_local_retrieval_repair.py"
        spec = importlib.util.spec_from_file_location("retrieval_repair_progress_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with patch.object(runner, "save") as saved, patch("builtins.print"):
            runner.event(Path("/not-created"), "segment-encoding",
                         vector_directory="validation-vectors", rows=8192, total=240765)
        self.assertEqual(saved.call_args.args[1]["vector_directory"], "validation-vectors")


if __name__ == "__main__":
    unittest.main()
