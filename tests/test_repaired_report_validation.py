"""In-memory checks only; no model inference, files or cleanup."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/verify_repaired_report.py"
spec = importlib.util.spec_from_file_location("verify_report", path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class RepairedReportValidationTests(unittest.TestCase):
    def setUp(self):
        self.frozen = {name: {"fingerprint": name + "-hash"} for name in module.NAMES if name != "cohere_v3"}
        self.dedicated = {name: {"instance_id": name + "-host"} for name in module.NAMES if name not in ("cohere_v3", "teacher")}
        self.report = {"catalog_sha256": "catalog", "cohere_v4": {"status": "blocked"}, "models": {}}
        for name in module.NAMES:
            identity = {"fingerprint": name + "-hash", "active_models": [name], "resource_id": name + "-host"}
            if name == "teacher": identity.update(device="cuda", gpu_name="test GPU")
            if name == "cohere_v3": identity["resource_id"] = "bedrock:cohere.embed-english-v3"
            rows = [{"query_id": str(index), "metrics": {"recall_at_1": .1, "recall_at_3": .3,
                "recall_at_5": .5, "recall_at_10": 1., "mrr_at_10": .5},
                "top10": [{"corpus_id": str(p)} for p in range(10)]} for index in range(1127)]
            self.report["models"][name] = {"quality": {"catalog_sha256": "catalog", "summary": {"query_count": 1127}, "per_query": rows},
                "measurement": {"model": name, "server": identity, "expected_measured_requests": 160,
                    "valid_successful_measurements": 160, "samples": [{"client_round_trip_seconds": .01} for _ in range(160)], "failures": []}}

    def validate(self):
        module.validate_report(self.report, "catalog", self.frozen, self.dedicated, "client-host")

    def test_complete_consistent_report_accepted(self):
        self.validate()

    def test_incomplete_timing_rejected(self):
        self.report["models"]["teacher"]["measurement"]["samples"].pop()
        with self.assertRaises(ValueError): self.validate()

    def test_changed_query_order_rejected(self):
        self.report["models"]["cohere_v3"]["quality"]["per_query"].reverse()
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate_parent_rejected(self):
        self.report["models"]["plain_student"]["quality"]["per_query"][0]["top10"][1]["corpus_id"] = "0"
        with self.assertRaises(ValueError): self.validate()

    def test_changed_weights_rejected(self):
        self.report["models"]["custom_student"]["measurement"]["server"]["fingerprint"] = "changed"
        with self.assertRaises(ValueError): self.validate()

    def test_cpu_teacher_rejected(self):
        self.report["models"]["teacher"]["measurement"]["server"]["device"] = "cpu"
        with self.assertRaises(ValueError): self.validate()

    def test_wrong_static_host_rejected(self):
        self.report["models"]["potion_retrieval"]["measurement"]["server"]["resource_id"] = "another-host"
        with self.assertRaises(ValueError): self.validate()

    def test_errors_are_not_hidden(self):
        self.report["models"]["cohere_v3"]["measurement"]["failures"] = [{"error": "timeout"}]
        with self.assertRaises(ValueError): self.validate()


if __name__ == "__main__":
    unittest.main()
