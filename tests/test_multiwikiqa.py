"""In-memory only: no temporary directories, model downloads or file cleanup."""
import copy
import unittest
from pycon_model2vec_demo.multiwikiqa import chunk_article, prepare, score_single_gold


CONFIG = {"languages": ["tl", "en"], "chunks": {"max_characters": 12, "overlap_characters": 4},
          "split": {"development_fraction": 0.2, "seed": "fixture"}}


def row(language="tl", title="Page", context="alpha beta gamma delta", answer="beta", start=6):
    return {"id": f"https://{language}.wikipedia.org/wiki/{title}", "title": title,
            "context": context, "question": "What word follows alpha?",
            "answers": {"text": [answer], "answer_start": [start]}}


class MultiWikiQATests(unittest.TestCase):
    def test_full_rows_and_exact_duplicate_are_preserved(self):
        result = prepare({"tl": [row(), row()], "en": [row("en")]}, CONFIG)
        self.assertEqual(len(result["annotations"]), 3)
        self.assertEqual(len(result["candidate_queries"]), 2)
        self.assertEqual(result["audit"]["counts"]["reason_duplicate-qa-annotation_tl"], 1)
        self.assertIsNotNone(result["annotations"][1]["duplicate_of"])

    def test_blank_answer_is_recorded_not_repaired(self):
        result = prepare({"tl": [row(answer="", start=0)], "en": []}, CONFIG)
        self.assertEqual(len(result["annotations"]), 1)
        self.assertEqual(result["candidate_queries"], [])
        self.assertIn("empty-answer-reference", result["annotations"][0]["exclusion_reasons"])

    def test_invalid_answer_offset_is_explicit(self):
        result = prepare({"tl": [row(start=2)], "en": []}, CONFIG)
        self.assertIn("invalid-answer-offset", result["annotations"][0]["exclusion_reasons"])

    def test_one_parent_many_queries_share_partition(self):
        other = row()
        other["question"] = "A different question"
        result = prepare({"tl": [row(), other], "en": []}, CONFIG)
        queries = result["candidate_queries"]
        self.assertEqual(queries[0]["gold_parent_id"], queries[1]["gold_parent_id"])
        self.assertEqual(queries[0]["partition"], queries[1]["partition"])

    def test_identical_context_across_languages_does_not_cross_partition(self):
        result = prepare({"tl": [row()], "en": [row("en")]}, CONFIG)
        self.assertEqual(len(result["parents"]), 2)
        self.assertEqual(result["parents"][0]["component_id"], result["parents"][1]["component_id"])
        self.assertEqual(result["parents"][0]["partition"], result["parents"][1]["partition"])

    def test_conflicting_article_context_stops(self):
        other = row(context="changed context", answer="changed", start=0)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            prepare({"tl": [row(), other], "en": []}, CONFIG)

    def test_chunking_uses_source_offsets_and_preserves_every_character(self):
        text = "á漢🙂 alpha\n" * 9
        parent = {"parent_id": "p", "language": "tl", "text": text}
        chunks = chunk_article(parent, size=12, overlap=4)
        positions = set()
        for chunk in chunks:
            self.assertEqual(chunk["text"], text[chunk["start"]:chunk["end"]])
            self.assertLessEqual(len(chunk["text"]), 12)
            positions.update(range(chunk["start"], chunk["end"]))
        self.assertEqual(positions, set(range(len(text))))

    def test_answer_locations_cannot_change_chunk_boundaries(self):
        a = row()
        b = row(answer="alpha", start=0)
        first = prepare({"tl": [a], "en": []}, CONFIG)
        second = prepare({"tl": [b], "en": []}, CONFIG)
        self.assertEqual(first["chunks"], second["chunks"])

    def test_supporting_chunk_and_pending_semantics(self):
        result = prepare({"tl": [row()], "en": []}, CONFIG)
        query = result["candidate_queries"][0]
        self.assertTrue(query["answer_bearing_chunk_ids"])
        self.assertEqual(query["semantic_review"], "pending")
        self.assertFalse(result["audit"]["model_scoring_ready"])

    def test_source_recall_does_not_prove_delivered_answer(self):
        query = {"gold_parent_id": "p", "answer_spans": [{"start": 20, "end": 24}],
                 "exclusion_reasons": []}
        result = score_single_gold(query, ["p"], {"p": {"parent_id": "p", "start": 0, "end": 12}})
        self.assertEqual(result["Recall@10"], 1)
        self.assertEqual(result["AnswerSpanCoverage@10"], 0)
        result = score_single_gold(query, ["p"], {"p": {"parent_id": "p", "start": 16, "end": 28}})
        self.assertEqual(result["AnswerSpanCoverage@1"], 1)

    def test_rank_cutoffs_never_exceed_ten(self):
        query = {"gold_parent_id": "gold", "answer_spans": [{"start": 1, "end": 2}]}
        result = score_single_gold(query, ["a", "b", "c", "gold"], {})
        self.assertEqual(result["Recall@3"], 0)
        self.assertEqual(result["Recall@5"], 1)
        self.assertEqual(result["MRR@10"], 0.25)
        with self.assertRaises(ValueError):
            score_single_gold(query, [str(i) for i in range(11)], {})
        with self.assertRaises(ValueError):
            score_single_gold(query, ["gold", "gold"], {})

    def test_wrong_parent_evidence_is_rejected(self):
        query = {"gold_parent_id": "p", "answer_spans": [{"start": 1, "end": 2}]}
        with self.assertRaises(ValueError):
            score_single_gold(query, ["p"], {"p": {"parent_id": "other", "start": 0, "end": 4}})

    def test_preparation_is_deterministic(self):
        rows = {"tl": [row()], "en": [row("en")]}
        self.assertEqual(prepare(rows, CONFIG), prepare(copy.deepcopy(rows), CONFIG))


if __name__ == "__main__":
    unittest.main()
