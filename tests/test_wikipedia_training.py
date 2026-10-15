"""Pure in-memory tests; no temporary directories or cleanup."""
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch
from pycon_model2vec_demo.wikipedia_training import (
    article_key, url_key, normalize, ContentGuard, passage_windows,
    article_passages, HashSample,
)

POLICY = {"paragraph_minimum_characters": 160, "snippet_words": 32,
          "snippet_stride_words": 16, "snippet_minimum_characters": 120}
SAMPLING = {"seed": "fixture", "maximum_passage_characters": 256,
            "minimum_passage_characters": 100, "maximum_passages_per_article": 3}


class WikipediaTrainingTests(unittest.TestCase):
    def test_url_title_encoding_and_underscore_alias(self):
        self.assertEqual(url_key("https://tl.wikipedia.org/wiki/San_Jos%C3%A9#x"),
                         article_key("tl", "San José"))
        self.assertNotEqual(article_key("tl", "Python"), article_key("en", "Python"))

    def test_non_wikipedia_url_rejected(self):
        with self.assertRaises(ValueError):
            url_key("https://example.com/wiki/foo")

    def test_normalization_preserves_words_not_punctuation_equivalence(self):
        self.assertEqual(normalize("  ＡLPHA\n beta "), "alpha beta")
        self.assertNotEqual(normalize("alpha,beta"), normalize("alpha beta"))

    def test_entire_context_overlap(self):
        guard = ContentGuard([{"parent_id": "gold", "text": "alpha beta"}], POLICY)
        self.assertEqual(guard.find("ALPHA\n  beta"), ("normalized-full-context", "gold"))

    def test_paragraph_copied_into_other_article(self):
        para = "long benchmark paragraph with specific unusual terminology " * 6
        guard = ContentGuard([{"parent_id": "gold", "text": para + "\n\nsecond part"}], POLICY)
        self.assertEqual(guard.find("new introduction\n\n" + para), ("exact-long-paragraph", "gold"))

    def test_snippet_found_at_different_offset(self):
        benchmark = " ".join(f"unusualword{i}" for i in range(90))
        guard = ContentGuard([{"parent_id": "gold", "text": benchmark}], POLICY)
        candidate = "new words " + " ".join(benchmark.split()[16:48]) + " extra words"
        self.assertEqual(guard.find(candidate), ("sampled-exact-32-word-snippet", "gold"))

    def test_common_short_phrase_not_called_leakage(self):
        guard = ContentGuard([{"parent_id": "gold", "text": "The capital of the country"}], POLICY)
        self.assertIsNone(guard.find("The capital of the country has several attractions."))

    def test_training_windows_keep_original_offsets_and_no_overlap(self):
        text = "á漢🙂 alpha beta gamma\n" * 80
        windows = list(passage_windows(text, 256, 100))
        for window in windows:
            self.assertEqual(window["text"], text[window["start"]:window["end"]])
            self.assertLessEqual(len(window["text"]), 256)
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(windows, windows[1:])))

    def test_short_tail_not_called_full_coverage(self):
        windows = list(passage_windows("x" * 300, 256, 100))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["end"], 256)

    def test_per_article_cap_and_no_question_fields(self):
        row = {"id": "42", "url": "https://tl.wikipedia.org/wiki/Example",
               "title": "Example", "text": "alpha beta gamma " * 1000}
        selected, eligible = article_passages("tl", row, "source", 1, SAMPLING)
        self.assertGreater(eligible, 3)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all("question" not in value and "answer" not in value for _, value in selected))

    def test_bottom_hash_sample_independent_of_row_order(self):
        sample_a, sample_b = HashSample(3), HashSample(3)
        pairs = [(i, {"passage_id": str(i), "text": "x"}) for i in [9, 4, 1, 8, 2]]
        for priority, record in pairs:
            sample_a.add(record, priority)
        for priority, record in reversed(pairs):
            sample_b.add(record, priority)
        self.assertEqual(sample_a.records(), sample_b.records())
        self.assertEqual([r["passage_id"] for r in sample_a.records()], ["1", "2", "4"])

    def test_bounds_invalid(self):
        with self.assertRaises(ValueError):
            list(passage_windows("text", 1, 2))

    def test_frozen_alias_translation_and_missing_page_exclusions(self):
        spec = importlib.util.spec_from_file_location(
            "wikipedia_preparer", Path(__file__).resolve().parents[1] / "scripts/prepare_wikipedia_training.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parents = [{"language": "tl", "title": "Lumang pangalan",
                    "source_url": "https://tl.wikipedia.org/wiki/Lumang_pangalan"}]
        responses = [{"language": "tl", "response": {"query": {
            "redirects": [{"from": "Lumang pangalan", "to": "Bagong pangalan"}],
            "pages": [{"title": "Bagong pangalan", "langlinks": [{"lang": "en", "title": "Translated page"}]},
                      {"title": "Deleted article", "missing": True}]}}}]
        keys, audit = module.exclusion_keys(parents, responses)
        self.assertIn(article_key("tl", "Lumang pangalan"), keys)
        self.assertIn(article_key("tl", "Bagong pangalan"), keys)
        self.assertIn(article_key("en", "Translated page"), keys)
        self.assertIn(article_key("tl", "Deleted article"), keys)
        self.assertEqual(audit["linked_translation_entries"], 1)
        self.assertEqual(len(audit["unresolved_live_pages"]), 1)

    def test_provider_33_second_backoff_is_honored_not_rejected(self):
        spec = importlib.util.spec_from_file_location(
            "wikipedia_preparer", Path(__file__).resolve().parents[1] / "scripts/prepare_wikipedia_training.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        limited = Mock(status_code=429, headers={"Retry-After": "33"})
        success = Mock(status_code=200)
        success.json.return_value = {"query": {"pages": [{"title": "Example"}]}}
        with patch.object(module.requests, "get", side_effect=[limited, success]), \
                patch.object(module.time, "sleep") as sleeper:
            result = module.resolve_links(("tl", ["Example"]))
            self.assertEqual(result["response"], success.json.return_value)
            self.assertEqual(sum(call.args[0] for call in sleeper.call_args_list), 33)


if __name__ == "__main__":
    unittest.main()
