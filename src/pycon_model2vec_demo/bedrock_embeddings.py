"""Cohere requests, verified response decoding, and bounded Bedrock calls."""

from __future__ import annotations

from collections import deque
import json
import threading
import time
from typing import Mapping, Sequence

import numpy as np


def request_body(texts: Sequence[str], role: str, model: Mapping[str, object]) -> dict:
    if role not in ("search_query", "search_document"):
        raise ValueError("Embedding role must be search_query or search_document")
    if not 1 <= len(texts) <= 96 or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Cohere needs one to 96 nonempty texts")
    limit = model.get("maximum_input_characters")
    if model.get("full_content") and limit and any(len(text) > int(limit) for text in texts):
        raise ValueError("Full-content input exceeds character limit; segment before invoking Bedrock")
    if model.get("full_content") and model["truncate"] != "NONE":
        raise ValueError("Full-content encoding requires provider truncate=NONE")
    sent_texts = [text[:int(limit)] for text in texts] if limit else list(texts)
    body = {"texts": sent_texts, "input_type": role, "embedding_types": ["float"], "truncate": model["truncate"]}
    if "embed-v4" in str(model["model_id"]):
        body["output_dimension"] = model["dimension"]
    return body


def decode_vectors(response: Mapping[str, object], texts: Sequence[str], dimension: int) -> np.ndarray:
    # v3 echoes tokenizer-normalized text (lowercase with spaced punctuation),
    # and may echo truncated text. The API defines vector order as input order;
    # echo strings cannot be compared byte-for-byte to the original inputs.
    if "texts" in response and len(response["texts"]) != len(texts):
        raise ValueError("Cohere response text count differs from request")
    values = response.get("embeddings")
    if isinstance(values, dict):
        values = values.get("float")
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (len(texts), dimension) or not np.isfinite(vectors).all():
        raise ValueError("Cohere returned an invalid vector shape or nonfinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cohere returned a zero vector")
    return vectors / norms


class TokenWindow:
    """Reserve estimated tokens; replace them with Bedrock's actual token count."""

    def __init__(self, maximum: int):
        self.maximum = maximum
        self.events: deque[list[float]] = deque()
        self.lock = threading.Lock()
        self.token_char_ratio = 0.30

    def reserve(self, characters: int) -> list[float]:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and self.events[0][0] <= now - 61:
                    self.events.popleft()
                estimate = max(1, int(characters * self.token_char_ratio * 1.05))
                if estimate > self.maximum:
                    raise ValueError("One request exceeds the token-window budget")
                if sum(event[1] for event in self.events) + estimate <= self.maximum:
                    event = [now, float(estimate)]
                    self.events.append(event)
                    return event
                delay = max(0.05, min(1.0, self.events[0][0] + 61 - now))
            time.sleep(delay)

    def reconcile(self, event: list[float], tokens: int | None, characters: int) -> None:
        with self.lock:
            if tokens is not None:
                event[1] = float(tokens)
                if characters:
                    observed = tokens / characters
                    self.token_char_ratio = 0.9 * self.token_char_ratio + 0.1 * observed


class BedrockCohere:
    def __init__(self, client: object, model: Mapping[str, object], *, tokens_per_minute: int = 285000):
        self.client = client
        self.model = model
        self.window = TokenWindow(tokens_per_minute)

    def encode(self, texts: Sequence[str], role: str) -> tuple[np.ndarray, dict]:
        from botocore.exceptions import ClientError, ConnectionClosedError, EndpointConnectionError, ReadTimeoutError

        body = request_body(texts, role, self.model)
        characters = sum(map(len, body["texts"]))
        failures = []
        started = time.perf_counter()
        for attempt in range(8):
            event = self.window.reserve(characters)
            call_started = time.perf_counter()
            try:
                result = self.client.invoke_model(modelId=self.model["model_id"], body=json.dumps(body), contentType="application/json", accept="application/json")
                payload = json.loads(result["body"].read())
                call_seconds = time.perf_counter() - call_started
                headers = result["ResponseMetadata"]["HTTPHeaders"]
                tokens = headers.get("x-amzn-bedrock-input-token-count")
                tokens = int(tokens) if tokens is not None else None
                self.window.reconcile(event, tokens, characters)
                vectors = decode_vectors(payload, texts, int(self.model["dimension"]))
                return vectors, {
                    "request_id": result["ResponseMetadata"]["RequestId"],
                    "response_id": payload.get("id"),
                    "input_type": role,
                    "texts": len(texts),
                    "provider_transformed_echo_texts": sum(before != after for before, after in zip(body["texts"], payload.get("texts", body["texts"]))),
                    "character_truncated_inputs": sum(before != after for before, after in zip(texts, body["texts"])),
                    "original_characters": sum(map(len, texts)),
                    "characters": characters,
                    "input_tokens": tokens,
                    "service_latency_ms": int(headers["x-amzn-bedrock-invocation-latency"]) if "x-amzn-bedrock-invocation-latency" in headers else None,
                    "api_seconds": call_seconds,
                    "total_seconds_including_waits": time.perf_counter() - started,
                    "retries": failures,
                }
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code not in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "InternalServerException", "ModelTimeoutException"):
                    raise
                failures.append({"code": code, "message": exc.response["Error"]["Message"]})
                if code == "ThrottlingException":
                    self.window.maximum = max(50000, int(self.window.maximum * 0.95))
            except (ConnectionClosedError, EndpointConnectionError, ReadTimeoutError) as exc:
                failures.append({"code": type(exc).__name__, "message": str(exc)})
            time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"Bedrock encoding failed after retries: {failures}")
