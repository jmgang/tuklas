import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycon_model2vec_demo.data import load_disastir_slice, load_qa_collection


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.query_path = root / "QA_meteorological_hydrological.json"
        self.qrels_path = root / "QA_meteorological_hydrological_qrels.json"
        self.queries = [{"user_query": f"Flood question {i}?"} for i in range(20)]
        self.qrels = [
            {"user_query": query["user_query"], "passage": f"Positive passage {i}", "score": 2}
            for i, query in enumerate(self.queries)
        ] + [
            {"user_query": query["user_query"], "passage": f"Negative passage {i}", "score": 0}
            for i, query in enumerate(self.queries)
        ]
        self.write_files()

    def write_files(self):
        self.query_path.write_text(json.dumps(self.queries), encoding="utf-8")
        self.qrels_path.write_text(json.dumps(self.qrels), encoding="utf-8")

    def load(self, **overrides):
        options = {
            "query_path": self.query_path,
            "qrels_path": self.qrels_path,
            "revision": "pinned-test-revision",
            "task": "QA",
            "category": "meteorological_hydrological",
            "seed": "test-seed",
            "evaluation_fraction": 0.25,
        }
        options.update(overrides)
        return load_disastir_slice(**options)

    def test_loads_and_freezes_disjoint_groups(self):
        prepared = self.load()
        repeated = self.load()
        self.assertEqual(prepared, repeated)
        self.assertEqual(prepared.manifest["query_count"], 20)
        self.assertEqual(prepared.manifest["judged_pair_count"], 40)
        self.assertEqual(prepared.manifest["score_counts"][2], 20)
        self.assertFalse(
            set(prepared.manifest["development_query_ids"])
            & set(prepared.manifest["evaluation_query_ids"])
        )
        self.assertEqual(
            len(prepared.manifest["development_query_ids"])
            + len(prepared.manifest["evaluation_query_ids"]),
            20,
        )

    def test_rejects_unexpected_score(self):
        self.qrels[0]["score"] = 4
        self.write_files()
        with self.assertRaisesRegex(ValueError, "score must be an integer from 0 to 3"):
            self.load()

    def test_rejects_unknown_query(self):
        self.qrels[0]["user_query"] = "Not in queries"
        self.write_files()
        with self.assertRaisesRegex(ValueError, "query not found"):
            self.load()

    def test_reports_and_excludes_orphan_judgment_when_requested(self):
        self.qrels.append(
            {"user_query": "Unpublished question?", "passage": "Orphan passage", "score": 2}
        )
        self.write_files()
        prepared = self.load(orphan_qrels_policy="report_exclude")
        self.assertEqual(prepared.manifest["raw_judgment_count"], 41)
        self.assertEqual(prepared.manifest["judged_pair_count"], 40)
        self.assertEqual(prepared.manifest["excluded_orphan_judgment_count"], 1)
        self.assertEqual(prepared.manifest["orphan_query_count"], 1)

    def test_rejects_duplicate_query_passage(self):
        self.qrels.append(dict(self.qrels[0]))
        self.write_files()
        with self.assertRaisesRegex(ValueError, "Candidate IDs must be unique"):
            self.load()

    def test_rejects_bad_checksum(self):
        with self.assertRaisesRegex(ValueError, "Query file SHA-256 differs"):
            self.load(expected_query_sha256="0" * 64)

    def test_rejects_query_without_judgments(self):
        self.qrels = [row for row in self.qrels if row["user_query"] != "Flood question 0?"]
        self.write_files()
        with self.assertRaisesRegex(ValueError, "no judged candidates"):
            self.load()

    def test_rejects_non_qa_task(self):
        with self.assertRaisesRegex(ValueError, "Only DisastIR QA"):
            self.load(task="NLI")

    def test_combines_categories_without_cross_split_leakage(self):
        other_query = self.query_path.with_name("QA_geohazard.json")
        other_qrels = self.qrels_path.with_name("QA_geohazard_qrels.json")
        other_query.write_text(
            json.dumps([{"user_query": f"Landslide question {i}?"} for i in range(20)]),
            encoding="utf-8",
        )
        other_qrels.write_text(
            json.dumps(
                [
                    {
                        "user_query": f"Landslide question {i}?",
                        "passage": f"Landslide passage {i}",
                        "score": 1,
                    }
                    for i in range(20)
                ]
            ),
            encoding="utf-8",
        )
        checksums = {
            "meteorological_hydrological": {
                "query_sha256": hashlib.sha256(self.query_path.read_bytes()).hexdigest(),
                "qrels_sha256": hashlib.sha256(self.qrels_path.read_bytes()).hexdigest(),
            },
            "geohazard": {
                "query_sha256": hashlib.sha256(other_query.read_bytes()).hexdigest(),
                "qrels_sha256": hashlib.sha256(other_qrels.read_bytes()).hexdigest(),
            },
        }
        prepared = load_qa_collection(
            data_dir=Path(self.temp.name),
            revision="pinned-test-revision",
            category_checksums=checksums,
            seed="test-seed",
            evaluation_fraction=0.75,
        )
        self.assertEqual(prepared.manifest["query_count"], 40)
        self.assertEqual(prepared.manifest["judged_pair_count"], 60)
        self.assertFalse(
            set(prepared.manifest["development_query_ids"])
            & set(prepared.manifest["evaluation_query_ids"])
        )

    def test_manifest_records_actual_checksums(self):
        prepared = self.load()
        expected = hashlib.sha256(self.qrels_path.read_bytes()).hexdigest()
        self.assertEqual(prepared.manifest["qrels_sha256"], expected)

    def test_accepts_compressed_source_with_original_checksum(self):
        compressed = self.qrels_path.with_suffix(".json.gz")
        compressed.write_bytes(gzip.compress(self.qrels_path.read_bytes()))
        expected = hashlib.sha256(self.qrels_path.read_bytes()).hexdigest()
        prepared = self.load(qrels_path=compressed, expected_qrels_sha256=expected)
        self.assertEqual(prepared.manifest["judged_pair_count"], 40)


if __name__ == "__main__":
    unittest.main()
