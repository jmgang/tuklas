import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.vocabulary import load_stopwords, normalize_tokens, passage_terms, select_vocabulary


class VocabularyTests(unittest.TestCase):
    def test_unicode_casefold_and_alpha_only(self):
        self.assertEqual(normalize_tokens("ＦＬＯＯＤ №42 Côte-d'Azur A e"), ["flood", "no", "côte", "azur"])

    def test_distinct_passage_counts_and_lexical_tie(self):
        stopwords = frozenset({"the", "of"})
        selected, report = select_vocabulary(
            ["The flood flood water", "Flood water", "Water landslide"],
            stopwords=stopwords, ngram_lengths=(1, 2, 3), min_passage_frequency=2, candidate_cap=3,
        )
        self.assertEqual(selected, [("water", 3), ("flood", 2), ("flood water", 2)])
        self.assertEqual(report["passage_count"], 3)
        self.assertNotIn("the", passage_terms("the of", stopwords, (1, 2)))

    def test_versioned_stopwords_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stopwords.txt"
            path.write_text("# v1\nthe\nof\n", encoding="utf-8")
            words, digest = load_stopwords(path)
            self.assertEqual(words, frozenset({"the", "of"}))
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
