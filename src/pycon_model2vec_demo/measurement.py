"""Operational measures for an isolated, one-model Python process."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import resource
from statistics import median
import time
from typing import Callable, Sequence

import numpy as np

from .encoders import RetrievalEncoder
from .retrieval import exact_top_k


def artifact_size_bytes(directory: Path) -> int:
    """Size of saved model files, excluding no files inside the artifact."""
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def peak_rss_bytes() -> int:
    """Process high-water RSS; macOS reports bytes, Linux reports KiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _timing_summary(samples_seconds: Sequence[float]) -> dict[str, object]:
    if not samples_seconds or any(sample < 0 for sample in samples_seconds):
        raise ValueError("Timing samples must be nonempty and nonnegative")
    ms = np.asarray(samples_seconds, dtype=np.float64) * 1000
    return {
        "repetitions": len(ms),
        "median_ms": float(median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
    }


def repeated_timing(operation: Callable[[], object], *, warmup: int = 2, repetitions: int = 7) -> dict[str, object]:
    if warmup < 0 or repetitions <= 0:
        raise ValueError("Warmup must be nonnegative and repetitions positive")
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return _timing_summary(samples)


def measure_warm_workload(
    encoder: RetrievalEncoder,
    *,
    query_texts: Sequence[str],
    passage_texts: Sequence[str],
    passage_vectors: np.ndarray,
    corpus_ids: Sequence[str],
    top_k: int = 10,
    search_block_size: int = 4096,
    warmup: int = 2,
    repetitions: int = 5,
) -> dict[str, object]:
    """Repeated warm CPU measurements, using the same texts and corpus for each model.

    Query samples are single-request, not a batched average. Passage encoding
    uses one fixed batch, while exact search uses already-built corpus vectors.
    """
    if not query_texts or not passage_texts:
        raise ValueError("Warm workload needs both queries and passages")
    if len(passage_vectors) != len(corpus_ids):
        raise ValueError("Corpus vectors and IDs differ")
    for _ in range(warmup):
        encoder.encode_passages(passage_texts)
        for text in query_texts:
            query_vector = encoder.encode_queries([text])
            exact_top_k(query_vector, passage_vectors, corpus_ids, k=top_k, corpus_block_size=search_block_size)

    passage_samples: list[float] = []
    query_encode_samples: list[float] = []
    exact_search_samples: list[float] = []
    end_to_end_samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        encoder.encode_passages(passage_texts)
        passage_samples.append(time.perf_counter() - started)
        for text in query_texts:
            started = time.perf_counter()
            vector = encoder.encode_queries([text])
            encoded = time.perf_counter()
            exact_top_k(vector, passage_vectors, corpus_ids, k=top_k, corpus_block_size=search_block_size)
            searched = time.perf_counter()
            query_encode_samples.append(encoded - started)
            exact_search_samples.append(searched - encoded)
            end_to_end_samples.append(searched - started)
    passage_summary = _timing_summary(passage_samples)
    passage_summary["texts_per_batch"] = len(passage_texts)
    passage_summary["median_texts_per_second"] = len(passage_texts) / (passage_summary["median_ms"] / 1000)
    return {
        "query_count": len(query_texts),
        "passage_batch_size": len(passage_texts),
        "warmup_rounds": warmup,
        "repeated_rounds": repetitions,
        "passage_batch_encoding": passage_summary,
        "single_query_encoding": _timing_summary(query_encode_samples),
        "single_query_exact_search": _timing_summary(exact_search_samples),
        "single_query_end_to_end": _timing_summary(end_to_end_samples),
        "peak_process_rss_bytes": peak_rss_bytes(),
    }


def host_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
