"""One ephemeral GPU job: pinned SEA-LION -> two official TokenLearn students.

All durable objects are new S3 keys. No benchmark questions/answers are staged.
The managed training instance is released by SageMaker on success or failure.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import gc
import gzip
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from uuid import uuid4

INPUT = Path(os.environ.get("SEALION_INPUT_DIR", "/opt/ml/input/data/input"))
WORK = Path(os.environ.get("SEALION_WORK_DIR", "/opt/ml/sealion-work"))
# Packaging only: reconstruct exact original bytes, preserving the gzip inputs.
# Logical input SHA256 checks below still reference the original dataset files.
for compressed_input in INPUT.glob("*.jsonl.gz"):
    destination = compressed_input.with_suffix("")
    with gzip.open(compressed_input, "rb") as source, destination.open("xb") as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
CONFIG = json.loads((INPUT / "settings.json").read_text())


def digest(path):
    result = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")


def bootstrap():
    # Keep the DLC's CUDA-enabled torch. Pin actual experiment dependencies.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                    "model2vec[distill,train]==0.9.0", "sentence-transformers==5.7.0",
                    "transformers==5.3.0", "huggingface-hub==1.33.0", "datasets==5.0.1",
                    "numpy==2.5.3", "boto3==1.43.108"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
                    "tokenlearn @ git+https://github.com/MinishLab/tokenlearn.git@" +
                    CONFIG["tokenlearn"]["commit"]], check=True)


class Preservation:
    def __init__(self):
        import boto3
        self.s3 = boto3.client("s3", region_name="ap-southeast-1")
        self.bucket = os.environ["SEALION_OUTPUT_BUCKET"]
        self.prefix = os.environ["SEALION_OUTPUT_PREFIX"]
        self.index = {}

    def put(self, path, relative):
        path = Path(path)
        checksum = digest(path)
        key = f"{self.prefix}/{relative}"
        with path.open("rb") as source:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=source,
                               ContentLength=path.stat().st_size, IfNoneMatch="*",
                               Metadata={"sha256": checksum}, ServerSideEncryption="AES256")
        head = self.s3.head_object(Bucket=self.bucket, Key=key)
        if head["ContentLength"] != path.stat().st_size or head["Metadata"].get("sha256") != checksum:
            raise ValueError("S3 preservation metadata mismatch")
        self.index[relative] = {"sha256": checksum, "bytes": path.stat().st_size}

    def event(self, stage, **detail):
        value = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **detail}
        print(json.dumps(value, ensure_ascii=False), flush=True)
        path = WORK / "events" / f"{time.time_ns()}-{uuid4().hex[:6]}.json"
        write_json(path, value)
        self.put(path, "events/" + path.name)

    def tree(self, directory, relative):
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                self.put(path, relative + "/" + path.relative_to(directory).as_posix())


def prepare_arm(name, tokenizer, preservation):
    from sealion_training import article_partition, safe_windows
    arm = CONFIG["arms"][name]
    parents = {}
    if name == "approved_articles":
        with (INPUT / "parents.jsonl").open() as source:
            parents = {row["parent_id"]: row for row in map(json.loads, source)}
        names = ["chunks.candidate.jsonl"]
    else:
        names = ["training.en.jsonl", "training.tl.jsonl"]
    rows = []
    sources = 0
    split_sources = 0
    counts = Counter()
    for filename in names:
        with (INPUT / filename).open() as source:
            for original in map(json.loads, source):
                sources += 1
                text = original["text"]
                if sha256(text.encode()).hexdigest() != original.get("text_sha256", sha256(text.encode()).hexdigest()):
                    raise ValueError("Source text checksum mismatch")
                if name == "approved_articles":
                    parent = parents[original["parent_id"]]
                    # Exact shared article contexts stay in one validation partition.
                    article_id = parent["component_id"]
                    source_id = original.get("chunk_id", original.get("passage_id"))
                else:
                    article_id = original["source_url"]
                    source_id = original["passage_id"]
                if source_id is None:
                    raise ValueError("Missing source identity")
                partition = article_partition(article_id)
                windows = safe_windows(text, tokenizer)
                split_sources += int(len(windows) > 1)
                for start, end, fragment in windows:
                    row = {"text": fragment, "article_id": article_id, "source_id": source_id,
                           "source_start": start, "source_end": end,
                           "language": original["language"], "partition": partition,
                           "text_sha256": sha256(fragment.encode()).hexdigest(),
                           "token_count_with_specials": len(tokenizer.encode(fragment, add_special_tokens=True))}
                    rows.append(row)
                    counts[f"{partition}_{row['language']}"] += 1
    if not rows or not all(any(r["partition"] == p for r in rows) for p in ("train", "validation")):
        raise ValueError("Missing training/validation partition")
    # Batch neighboring lengths together for teacher encoding; alignment is frozen.
    rows.sort(key=lambda r: (r["token_count_with_specials"], r["article_id"], r["source_id"], r["source_start"]))
    train_articles = {r["article_id"] for r in rows if r["partition"] == "train"}
    val_articles = {r["article_id"] for r in rows if r["partition"] == "validation"}
    if train_articles & val_articles:
        raise ValueError("Article split leakage")
    path = WORK / f"{name}-training-layout.jsonl"
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    preservation.put(path, "layouts/" + path.name)
    preservation.event("training_layout_complete", arm=name, source_rows=sources,
                       training_windows=len(rows), extra_split_source_rows=split_sources,
                       partition_language_counts=dict(counts), training_articles=len(train_articles),
                       validation_articles=len(val_articles), layout_sha256=digest(path))
    return rows


def featurize_arm(name, rows, teacher, preservation):
    import numpy as np
    import tokenlearn.featurize as official
    directory = WORK / (name + "-features")
    directory.mkdir(exist_ok=False)
    old_truncate = official._truncate_text
    # Preflight already split every document. Keep original source spelling rather
    # than decode/re-encode it or trim trailing whitespace. No truncation is allowed.
    def checked_text(tokenizer, text):
        if len(tokenizer.encode(text, add_special_tokens=True)) > 512:
            raise ValueError("Teacher truncation would occur")
        return text
    official._truncate_text = checked_text
    started = time.monotonic()
    texts, vectors = [], []
    count = 0
    try:
        # Actual TokenLearn token-embedding mean algorithm, not a substitute target.
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            encoded = list(official._encode([r["text"] for r in batch], teacher, teacher.tokenizer))
            if len(encoded) != len(batch):
                raise ValueError("Teacher output row count mismatch")
            for row, (text, vector) in zip(batch, encoded):
                if text != row["text"] or vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError("Teacher feature alignment/dimension/nonfinite failure")
                texts.append(text)
                vectors.append(vector)
            count += len(batch)
            if len(texts) >= 1024 or count == len(rows):
                part = count - len(texts)
                text_path = directory / f"features-{part:08d}.json"
                vector_path = directory / f"features-{part:08d}.npy"
                write_json(text_path, texts)
                with vector_path.open("xb") as target:
                    np.save(target, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
                preservation.put(text_path, f"features/{name}/{text_path.name}")
                preservation.put(vector_path, f"features/{name}/{vector_path.name}")
                elapsed = time.monotonic() - started
                preservation.event("teacher_features_progress", arm=name, rows_done=count,
                                   rows_total=len(rows), elapsed_seconds=elapsed, rows_per_second=count / elapsed)
                texts, vectors = [], []
    finally:
        official._truncate_text = old_truncate
    return directory


def train_arm(name, rows, features, initial, preservation):
    import numpy as np
    import torch
    from datasets import Dataset, DatasetDict
    from model2vec import StaticModel
    from sealion_training import official_training_adapter
    texts, matrices = [], []
    for path in sorted(features.glob("features-*.json")):
        texts.extend(json.loads(path.read_text()))
        matrices.append(np.load(path.with_suffix(".npy"), allow_pickle=False))
    vectors = np.concatenate(matrices, axis=0)
    del matrices
    if texts != [r["text"] for r in rows] or vectors.shape != (len(rows), 1024):
        raise ValueError("Feature reconstruction lost alignment")
    val_mask = np.asarray([r["partition"] == "validation" for r in rows])
    val_texts = [text for text, val in zip(texts, val_mask) if val]
    val_vectors = vectors[val_mask]
    training_texts = [text for text, val in zip(texts, val_mask) if not val]
    training_vectors = vectors[~val_mask]
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite feature; refusing library filtering")
    data_path = WORK / (name + "-hf-features")
    if data_path.exists():
        raise FileExistsError(data_path)
    DatasetDict({"train": Dataset.from_dict({"text": training_texts, "embedding": training_vectors.tolist()})}).save_to_disk(str(data_path))
    del vectors, training_vectors, texts
    gc.collect()
    destination = WORK / name
    if destination.exists():
        raise FileExistsError(destination)
    validation_history = []
    def on_validation(metrics):
        validation_history.append(metrics)
        preservation.event("student_validation", arm=name, check=len(validation_history), **metrics)
    original_argv = sys.argv
    sys.argv = ["tokenlearn.train", "--model2vec-model-name", str(initial),
                "--data-path", str(data_path), "--data-split", "train", "--save-path", str(destination),
                "--device", "cuda", "--batch-size", "256", "--lr", "0.001", "--vocab-size", "0"]
    initialization_hashes = {p.name: digest(p) for p in initial.iterdir() if p.is_file()}
    started = time.monotonic()
    preservation.event("student_training_started", arm=name, training_rows=len(rows) - int(val_mask.sum()),
                       validation_rows=int(val_mask.sum()), initial_hashes=initialization_hashes,
                       pca_dims=None, target_pca_dims=None, dimension=1024,
                       training_entrypoint="tokenlearn.train", training_recipe=CONFIG["tokenlearn"])
    try:
        with official_training_adapter(val_texts, val_vectors, "<pad>", on_validation) as state:
            runpy.run_module("tokenlearn.train", run_name="__main__")
            pad_id = state["pad_id"]
    finally:
        sys.argv = original_argv
    result = StaticModel.from_pretrained(destination)
    if result.embedding.shape[1] != 1024 or not np.isfinite(result.embedding).all():
        raise ValueError("Student output dimension/nonfinite failure")
    if {p.name: digest(p) for p in initial.iterdir() if p.is_file()} != initialization_hashes:
        raise ValueError("Shared initialization was mutated")
    original = StaticModel.from_pretrained(initial)
    changed = int(np.any(result.embedding != original.embedding, axis=1).sum())
    if changed == 0 or not validation_history:
        raise ValueError("No evidence of TokenLearn training")
    samples = ["Ang Pilipinas ay isang bansa.", "The Philippines is an archipelago."]
    a = result.encode(samples)
    b = StaticModel.from_pretrained(destination).encode(samples)
    if not np.array_equal(a, b) or not np.isfinite(a).all():
        raise ValueError("Saved model reload parity failure")
    preservation.tree(destination, "models/" + name)
    report = {"arm": name, "dimension": 1024, "vocabulary_size": len(result.embedding),
              "pad_id": pad_id, "initial_hashes": initialization_hashes, "changed_token_rows": changed,
              "training_seconds": time.monotonic() - started, "validation_history": validation_history,
              "training_rows": len(rows) - int(val_mask.sum()), "validation_rows": int(val_mask.sum()),
              "reload_parity": True, "weights_precision": str(result.embedding.dtype),
              "no_pca": True, "no_qa_training_inputs": True,
              "training_recipe": CONFIG["tokenlearn"], "interpretation": CONFIG["arms"][name]["interpretation"]}
    path = WORK / (name + "-training-report.json")
    write_json(path, report)
    preservation.put(path, "reports/" + path.name)
    preservation.event("student_complete", arm=name, dimension=1024, changed_token_rows=changed,
                       validation_checks=len(validation_history), training_seconds=report["training_seconds"])
    del result, original
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    bootstrap()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from model2vec.distill import distill
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, str(INPUT))
    WORK.mkdir(exist_ok=False, parents=True)
    (WORK / "events").mkdir()
    preservation = Preservation()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required; refusing CPU fallback")
        torch.set_num_threads(8)
        torch.manual_seed(42)
        np.random.seed(42)
        runtime = {name: importlib.metadata.version(name) for name in
                   ("model2vec", "tokenlearn", "torch", "transformers", "sentence-transformers", "datasets", "numpy")}
        preservation.event("runtime_ready", libraries=runtime, gpu=torch.cuda.get_device_name(0),
                           tokenlearn_commit=CONFIG["tokenlearn"]["commit"])
        for arm in CONFIG["arms"].values():
            for filename, spec in arm["files"].items():
                if digest(INPUT / filename) != spec["sha256"]:
                    raise ValueError("Input checksum mismatch: " + filename)
        teacher_settings = CONFIG["teacher"]
        preservation.event("teacher_download_started", **teacher_settings)
        teacher_path = snapshot_download(teacher_settings["model_id"], revision=teacher_settings["revision"],
                                         ignore_patterns=["*.bin", "*.h5", "*.msgpack", "onnx/*", "openvino/*"])
        teacher = SentenceTransformer(teacher_path, device="cuda", trust_remote_code=False)
        teacher.eval()
        if teacher.get_sentence_embedding_dimension() != 1024 or teacher.max_seq_length != 512:
            raise ValueError("Pinned teacher architecture differs")
        if teacher.tokenizer.pad_token != "<pad>" or teacher.default_prompt_name is not None:
            raise ValueError("Unexpected padding or implicit teacher prompt")
        initial = WORK / "initialization"
        preservation.event("distillation_started", pca_dims=None, dimension=1024, precision="float32")
        student = distill(teacher_path, device="cuda", pca_dims=None, quantize_to="float32",
                         vocabulary=None, vocabulary_quantization=None, trust_remote_code=False)
        if student.embedding.shape[1] != 1024 or not np.isfinite(student.embedding).all():
            raise ValueError("Invalid initial student")
        student.save_pretrained(initial)
        preservation.tree(initial, "models/initialization")
        preservation.event("distillation_complete", shape=list(student.embedding.shape),
                           bytes=int(student.embedding.nbytes), actual_padding_id=student.tokenizer.token_to_id("<pad>"))
        del student
        gc.collect()
        torch.cuda.empty_cache()
        layouts, features = {}, {}
        for name in ("approved_articles", "external_wikipedia"):
            layouts[name] = prepare_arm(name, teacher.tokenizer, preservation)
            features[name] = featurize_arm(name, layouts[name], teacher, preservation)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        reports = []
        for name in ("approved_articles", "external_wikipedia"):
            reports.append(train_arm(name, layouts[name], features[name], initial, preservation))
        # Verify two distinct final models, not one model trained twice in sequence.
        hashes = [preservation.index[f"models/{name}/model.safetensors"]["sha256"]
                  for name in ("approved_articles", "external_wikipedia")]
        if len(set(hashes)) != 2 or reports[0]["initial_hashes"] != reports[1]["initial_hashes"]:
            raise ValueError("Students are not independent/shared-initialization arms")
        complete = {"status": "complete", "protocol": CONFIG, "runtime": runtime,
                    "training_reports": reports, "artifacts": dict(preservation.index),
                    "evaluation_status": "not run; semantic review and shared retrieval layout pending",
                    "compute_release": "SageMaker releases training compute after this process exits"}
        path = WORK / "completion.json"
        write_json(path, complete)
        preservation.put(path, "reports/completion.json")
        Path("/opt/ml/model").mkdir(parents=True, exist_ok=True)
        write_json(Path("/opt/ml/model/completion.json"), complete)
        preservation.event("complete", students=2, dimension=1024, pca=False,
                           completion_sha256=digest(path))
    except BaseException as exc:
        preservation.event("failed", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
