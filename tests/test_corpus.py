import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import Candidate, QueryGroup


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "corpus.jsonl.gz"
        rows = [
            {"corpus_id": "corpus_0", "text": "Flood warning now"},
            {"corpus_id": "corpus_1", "text": "Shelter available"},
            {"corpus_id": "corpus_2", "text": "Flood warning now"},
        ]
        self.path.write_bytes(gzip.compress("\n".join(json.dumps(row) for row in rows).encode()))

    def test_exact_whitespace_unmatched_and_duplicate_canonical_id(self):
        corpus = load_corpus(self.path)
        self.assertEqual(corpus.duplicate_text_count, 1)
        self.assertEqual(corpus.resolve("Flood warning now"), ("corpus_0", "exact"))
        group = QueryGroup(
            "q1", "Where?",
            (
                Candidate("a", "Flood warning now", 2),
                Candidate("b", "Shelter   available", 1),
                Candidate("c", "Unreleased document", 3),
            ),
            "geohazard",
        )
        mapped, report = map_query_groups([group], corpus)
        self.assertEqual(mapped[0].judgments, {"corpus_0": 2, "corpus_1": 1})
        self.assertEqual((report["exact_matches"], report["whitespace_matches"], report["unmatched"]), (1, 1, 1))
        self.assertEqual(report["unmatched_positive"], 1)

    def test_checksum_and_missing_positive(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_corpus(self.path, expected_sha256="0" * 64)
        expected = hashlib.sha256(gzip.decompress(self.path.read_bytes())).hexdigest()
        corpus = load_corpus(self.path, expected_sha256=expected)
        group = QueryGroup("q2", "Question", (Candidate("a", "Missing", 2),))
        _, report = map_query_groups([group], corpus)
        self.assertEqual(report["queries_without_mapped_positive"], 1)


if __name__ == "__main__":
    unittest.main()
