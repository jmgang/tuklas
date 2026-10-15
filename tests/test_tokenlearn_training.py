import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.tokenlearn_training import _feature_files, _passages


class TokenLearnPreparationTests(unittest.TestCase):
    def test_passages_add_only_teacher_passage_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as target:
                target.write(json.dumps({"corpus_id": "a", "text": "Flood warning"}) + "\n")
            self.assertEqual(list(_passages(source, "passage: ")), [{"text": "passage: Flood warning"}])

    def test_feature_shards_require_matching_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_0.json").write_text('["passage: flood"]', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _feature_files(root)
            np.save(root / "feature_0.npy", np.ones((1, 4), dtype=np.float32))
            self.assertEqual(_feature_files(root), [root / "feature_0.json"])


if __name__ == "__main__":
    unittest.main()
