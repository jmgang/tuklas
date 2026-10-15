import json
from pathlib import Path
from hashlib import sha256
import unittest

from pycon_model2vec_demo.sealion_source_integrity import validate_source_text
from pycon_model2vec_demo.wikipedia_training import text_hash

ROOT = Path(__file__).resolve().parents[1]


class SourceIntegrityTests(unittest.TestCase):
    def test_real_external_row_is_not_a_raw_text_hash(self):
        # The Tagalog half, not the English one: this project is Tagalog-only
        # (English records must not train, select, or be scored), and the two
        # files carry an identical row schema, so this is the better fixture.
        path = ROOT / "runs/multiwikiqa-training-text-20261004-v5/training.tl.jsonl"
        with path.open() as source:
            row = json.loads(next(source))
        self.assertNotEqual(sha256(row["text"].encode()).hexdigest(), row["text_sha256"])
        self.assertEqual(validate_source_text(row, "external_wikipedia"), row["text"])

    def test_external_hash_formula_matches_producer_and_preserves_original_text(self):
        text = "ＭＡＮＩＬＡ  Café\n\nPilipinas"
        row = {"text": text, "text_sha256": text_hash(text)}
        self.assertEqual(validate_source_text(row, "external_wikipedia"), text)

    def test_incorrect_external_identity_refused(self):
        with self.assertRaisesRegex(ValueError, "Normalized"):
            validate_source_text({"text": "Pilipinas", "text_sha256": "0"*64}, "external_wikipedia")

    def test_approved_text_is_literal(self):
        text = "Original CASE\n  spacing"
        row = {"text": text}
        self.assertEqual(validate_source_text(row, "approved_articles"), text)

    def test_raw_hash_check_not_relaxed_for_approved_arm(self):
        with self.assertRaisesRegex(ValueError, "Raw"):
            validate_source_text({"text": "text", "text_sha256": "0"*64}, "approved_articles")
