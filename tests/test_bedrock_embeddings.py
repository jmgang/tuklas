"""Offline checks for embedding roles, response order and grade-3 scoring."""

import unittest

import numpy as np

from pycon_model2vec_demo.bedrock_embeddings import decode_vectors, request_body
from pycon_model2vec_demo.grade3 import Grade3Query, TOP10_METRICS, score_query, score_top10_query, summarize
from pycon_model2vec_demo.retrieval import RankedPassage


class BedrockEmbeddingTests(unittest.TestCase):
    def test_query_and_document_use_the_provider_roles(self):
        model = {"model_id": "cohere.embed-english-v3", "dimension": 1024, "truncate": "END"}
        for role in ("search_query", "search_document"):
            body = request_body(["flood evacuation"], role, model)
            self.assertEqual(body["input_type"], role)
            self.assertEqual(body["texts"], ["flood evacuation"])
            self.assertNotIn("output_dimension", body)

    def test_v4_requests_its_native_dimension(self):
        body = request_body(["a question"], "search_query", {"model_id": "global.cohere.embed-v4:0", "dimension": 1536, "truncate": "RIGHT"})
        self.assertEqual(body["output_dimension"], 1536)

    def test_v3_character_limit_preserves_every_input_row(self):
        model = {"model_id": "cohere.embed-english-v3", "dimension": 1024, "truncate": "END", "maximum_input_characters": 2048}
        texts = ["f" * 4054, "short passage"]
        body = request_body(texts, "search_document", model)
        self.assertEqual(len(body["texts"]), 2)
        self.assertEqual(len(body["texts"][0]), 2048)
        self.assertEqual(body["texts"][1], texts[1])
        self.assertEqual(len(texts[0]), 4054)

    def test_invalid_batch_or_role_is_rejected(self):
        model = {"model_id": "cohere.embed-english-v3", "dimension": 1024, "truncate": "END"}
        for texts, role in (([], "search_query"), (["q"] * 97, "search_query"), ([""], "search_query"), (["q"], "query")):
            with self.assertRaises(ValueError):
                request_body(texts, role, model)

    def test_both_provider_embedding_formats_are_normalized(self):
        for values in ([[3, 4]], {"float": [[3, 4]]}):
            vector = decode_vectors({"embeddings": values, "texts": ["q"]}, ["q"], 2)
            np.testing.assert_allclose(vector, [[0.6, 0.8]], atol=1e-7)

    def test_invalid_vectors_and_wrong_echo_count_are_rejected(self):
        bad = [
            {"embeddings": [[0, 0]]},
            {"embeddings": [[float("nan"), 1]]},
            {"embeddings": [[1, 2, 3]]},
            {"embeddings": [[1, 2]], "texts": ["q", "extra"]},
        ]
        for response in bad:
            with self.assertRaises(ValueError):
                decode_vectors(response, ["q"], 2)

    def test_v3_normalized_echo_does_not_change_the_sent_text(self):
        original = "I'm interested in Floods."
        echo = "i ' m interested in floods ."
        vector = decode_vectors({"embeddings": [[3, 4]], "texts": [echo]}, [original], 2)
        np.testing.assert_allclose(vector, [[0.6, 0.8]], atol=1e-7)
        model = {"model_id": "cohere.embed-english-v3", "dimension": 1024, "truncate": "END"}
        self.assertEqual(request_body([original], "search_query", model)["texts"], [original])

    def test_top10_keeps_every_grade3_gold(self):
        query = Grade3Query("q", "question", "geohazard", 2, frozenset({"gold-a", "gold-b"}), frozenset({"gold-a", "gold-b"}))
        ranking = [RankedPassage("gold-a", 1.0)] + [RankedPassage(f"p{i}", 0.9) for i in range(9)]
        scores = score_top10_query(query, ranking)
        self.assertEqual(scores["recall_at_1"], 0.5)
        self.assertEqual(scores["recall_at_10"], 0.5)
        self.assertNotIn("recall_at_100", scores)
        old = score_query(query, ranking + [RankedPassage(f"tail{i}", 0.0) for i in range(90)])
        self.assertEqual(scores, {key: old[key] for key in TOP10_METRICS})

    def test_top10_rejects_duplicate_or_incomplete_rankings(self):
        query = Grade3Query("q", "question", "geohazard", 1, frozenset({"gold"}), frozenset({"gold"}))
        for ranking in ([RankedPassage("gold", 1.0)], [RankedPassage("gold", 1.0)] * 10):
            with self.assertRaises(ValueError):
                score_top10_query(query, ranking)

    def test_top10_bootstrap_accepts_only_the_reported_metrics(self):
        rows = [{"query_id": "q", "category": "geohazard", "metrics": {key: 0.5 for key in TOP10_METRICS}}]
        result = summarize(rows, repetitions=10, metrics=TOP10_METRICS)
        self.assertEqual(set(result["overall"]), set(TOP10_METRICS))
        self.assertNotIn("recall_at_100", result["bootstrap_95"]["metrics"])


if __name__ == "__main__":
    unittest.main()
