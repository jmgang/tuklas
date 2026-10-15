"""One-model serving and a transport-independent checked response format.

No datasets, corpus matrices or load generation are loaded by this module.
The SageMaker and private EC2 HTTP adapters use the same interface.
"""
from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import threading
import time

import numpy as np

from .encoders import _validate_vectors


def validate_request(request):
    if request.get("operation") == "identity":
        return None
    if request.get("operation") != "encode":
        raise ValueError("Unknown operation")
    if request.get("role") not in ("corpus", "queries", "development"):
        raise ValueError("Unknown text role")
    texts = request.get("texts")
    if not isinstance(texts, list) or not 1 <= len(texts) <= 1024:
        raise ValueError("Expected 1..1024 texts")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Text must be a nonempty string")
    return texts


def check_isolation(client_id, identity, expected_model, expected_fingerprint):
    if not client_id or not identity.get("resource_id"):
        raise ValueError("Missing compute identity")
    if client_id == identity["resource_id"]:
        raise ValueError("Load generator cannot share the serving resource")
    if identity.get("model") != expected_model or identity.get("fingerprint") != expected_fingerprint:
        raise ValueError("Wrong model or frozen weights")
    if identity.get("active_models") != [expected_model]:
        raise ValueError("Exactly one model must be active")
    if expected_model == "teacher" and (identity.get("device") != "cuda" or not identity.get("gpu_name")):
        raise ValueError("Teacher requires verified CUDA GPU serving")


def decode_response(response, count, dimension):
    if response.get("shape") != [count, dimension] or response.get("dtype") != "<f4":
        raise ValueError("Response shape/dtype mismatch")
    raw = base64.b64decode(response["vectors"], validate=True)
    if len(raw) != count * dimension * 4:
        raise ValueError("Response vector byte count mismatch")
    return _validate_vectors(np.frombuffer(raw, dtype="<f4").reshape(count, dimension), count, dimension)


class ModelServer:
    def __init__(self, root, name, resource_id, *, device="cpu", threads=4):
        root = Path(root)
        manifest = json.loads((root / "fresh-input-manifest.json").read_text())
        row = manifest["frozen_models"][name]
        path = root / row["path"]
        for relative, expected in row["files_sha256"].items():
            hasher = sha256()
            with (path / relative).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected:
                raise ValueError(f"Frozen artifact mismatch: {name}/{relative}")
        self.lock = threading.Lock()
        self.row = row
        self.identity = {"resource_id": resource_id, "model": name,
            "fingerprint": row["fingerprint"], "active_models": [name],
            "device": device, "threads": threads, "logical_cpus": os.cpu_count(),
            "platform": platform.platform(), "python": platform.python_version(),
            "model_bytes": sum((path / r).stat().st_size for r in row["files_sha256"])}
        self.name = name
        started = time.perf_counter()
        if name == "teacher":
            import torch
            from sentence_transformers import SentenceTransformer
            if device != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("No CPU fallback for E5 serving")
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            self.model = SentenceTransformer(str(path), device="cuda", local_files_only=True,
                                            trust_remote_code=False)
            self.model.float()
            self.model.max_seq_length = 512
            self.dimension = self.model.get_embedding_dimension()
            self.identity.update(gpu_name=torch.cuda.get_device_name(0),
                torch_version=torch.__version__, precision="float32", tf32=False)
            sample = ["query: What causes flash flooding?", "passage: Heavy rainfall can cause flash flooding."]
            gpu = self.model.encode(sample, normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
            self.model.to("cpu")
            cpu = self.model.encode(sample, normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
            self.model.to("cuda")
            cosine = np.sum(gpu * cpu, axis=1)
            if not np.allclose(gpu, cpu, atol=2e-4, rtol=2e-3) or float(cosine.min()) < .99999:
                raise ValueError("Fresh CPU/GPU correctness smoke failed")
            self.identity["fresh_cpu_gpu_smoke_min_cosine"] = float(cosine.min())
        else:
            from model2vec import StaticModel
            from threadpoolctl import threadpool_limits
            if device != "cpu":
                raise ValueError("Static deployment profile is CPU")
            self.thread_limit = threadpool_limits(limits=threads)
            self.model = StaticModel.from_pretrained(str(path))
            self.dimension = self.model.dim
            self.identity.update(precision="frozen artifact dtype")
        self.identity.update(dimension=self.dimension, cold_load_and_smoke_seconds=time.perf_counter()-started,
            libraries={n: version(n) for n in (["numpy", "sentence-transformers", "transformers"]
                       if name == "teacher" else ["numpy", "model2vec"])})

    def handle(self, request):
        texts = validate_request(request)
        if texts is None:
            return {"identity": self.identity}
        if request.get("expected_model") != self.name:
            raise ValueError("Requested model differs from isolated server")
        settings = self.row["settings"]
        prefix = settings["passage_prefix" if request["role"] == "corpus" else "query_prefix"]
        prefixed = [prefix + t for t in texts]
        # Validate lengths before invoking either library; neither may truncate.
        if self.name == "teacher":
            lengths = self.model.tokenizer(prefixed, truncation=False, add_special_tokens=True,
                                          return_length=True)["length"]
            if any(n > 512 for n in lengths):
                raise ValueError("Teacher input exceeds token budget")
        else:
            lengths = self.model.tokenize(prefixed, max_length=None)
            if any(len(ids) > 512 for ids in lengths):
                raise ValueError("Static input exceeds token budget")
            if any(len(t) > 512 * self.model.median_token_length for t in prefixed):
                raise ValueError("Static input exceeds character budget")
        with self.lock:
            import psutil
            if self.name == "teacher":
                import torch
                torch.cuda.synchronize()
            started = time.perf_counter()
            if self.name == "teacher":
                values = self.model.encode(prefixed, batch_size=32, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
                torch.cuda.synchronize()
            else:
                values = self.model.encode(prefixed, max_length=None, batch_size=1024,
                                           use_multiprocessing=False)
            seconds = time.perf_counter() - started
            values = _validate_vectors(values, len(texts), self.dimension).astype("<f4", copy=False)
            telemetry = {"encode_seconds": seconds, "server_rss_bytes": psutil.Process().memory_info().rss}
            if self.name == "teacher":
                telemetry.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                                 gpu_reserved_bytes=torch.cuda.memory_reserved())
            return {"identity": self.identity, "shape": list(values.shape), "dtype": "<f4",
                "vectors": base64.b64encode(values.tobytes()).decode("ascii"), "telemetry": telemetry}
