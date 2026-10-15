"""In-memory tests only: no directories, cleanup, model downloads or AWS calls."""
import base64
import unittest
import numpy as np
from pycon_model2vec_demo.isolated_serving import check_isolation, decode_response, validate_request
from pycon_model2vec_demo.isolated_client import RemoteEncoder, private_http_transport


class IsolatedServingTests(unittest.TestCase):
    def identity(self):
        return {"resource_id": "server", "model": "teacher", "fingerprint": "pinned",
                "active_models": ["teacher"], "device": "cuda", "gpu_name": "Tesla T4"}

    def test_separate_resource_is_accepted(self):
        check_isolation("client", self.identity(), "teacher", "pinned")

    def test_same_resource_refused(self):
        with self.assertRaises(ValueError):
            check_isolation("server", self.identity(), "teacher", "pinned")

    def test_cpu_teacher_refused(self):
        row = self.identity(); row["device"] = "cpu"
        with self.assertRaises(ValueError):
            check_isolation("client", row, "teacher", "pinned")

    def test_extra_model_refused(self):
        row = self.identity(); row["active_models"].append("plain_student")
        with self.assertRaises(ValueError):
            check_isolation("client", row, "teacher", "pinned")

    def test_wrong_weights_refused(self):
        with self.assertRaises(ValueError):
            check_isolation("client", self.identity(), "teacher", "changed")

    def test_missing_identity_refused(self):
        row = self.identity(); row.pop("resource_id")
        with self.assertRaises(ValueError):
            check_isolation("client", row, "teacher", "pinned")

    def test_vector_roundtrip_and_corrupt_shape(self):
        values = np.array([[.6, .8]], dtype="<f4")
        row = {"shape": [1, 2], "dtype": "<f4", "vectors": base64.b64encode(values.tobytes()).decode()}
        np.testing.assert_array_equal(decode_response(row, 1, 2), values)
        with self.assertRaises(ValueError):
            decode_response(row, 2, 2)

    def test_bad_requests(self):
        for row in ({"operation": "encode", "role": "queries", "texts": [""]},
                    {"operation": "encode", "role": "unknown", "texts": ["text"]},
                    {"operation": "switch-model"},
                    {"operation": "encode", "role": "corpus", "texts": ["text"] * 1025}):
            with self.assertRaises(ValueError):
                validate_request(row)

    def test_normal_query_and_identity(self):
        self.assertEqual(validate_request({"operation": "encode", "role": "queries", "texts": ["rain"]}), ["rain"])
        self.assertIsNone(validate_request({"operation": "identity"}))

    def test_client_uses_remote_vectors_and_rejects_model_change(self):
        identity = {**self.identity(), "dimension": 2}
        def send(request):
            if request["operation"] == "identity":
                return {"identity": dict(identity)}
            return {"identity": dict(identity), "shape": [1, 2], "dtype": "<f4",
                "vectors": base64.b64encode(np.array([[.6, .8]], dtype="<f4").tobytes()).decode(),
                "telemetry": {"encode_seconds": .01}}
        encoder = RemoteEncoder(send, client_id="client", model="teacher", fingerprint="pinned")
        values, timing = encoder.encode(["rain"], "queries")
        self.assertEqual(values.shape, (1, 2))
        self.assertIn("client_round_trip_seconds", timing)
        identity["model"] = "other"
        with self.assertRaises(ValueError):
            encoder.encode(["rain"], "queries")

    def test_public_serving_route_refused(self):
        for url in ("http://8.8.8.8:8080/encode", "https://10.84.1.10/encode"):
            with self.assertRaises(ValueError):
                private_http_transport(url)

    def test_huggingface_hooks_return_json_body_not_tuple(self):
        import importlib.util
        import json
        from pathlib import Path
        path=Path(__file__).resolve().parents[1]/"scripts/isolated_sagemaker_inference.py"
        spec=importlib.util.spec_from_file_location("gpu_hooks",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        body=module.output_fn({"ready":True},"application/json")
        self.assertIsInstance(body,str)
        self.assertEqual(json.loads(body),{"ready":True})
        self.assertEqual(module.input_fn(body,"application/json"),{"ready":True})
