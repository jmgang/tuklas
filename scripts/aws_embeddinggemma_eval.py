#!/usr/bin/env python3
"""EmbeddingGemma-300m on the shared Tagalog benchmark: quality and runtime.

Runs on the same host type and protocol as the 20-system comparison:
  * runtime: the child from aws_runtime_comparison_v2.py (transformer on the
    T4 GPU, 591 timed single queries, full 40,152-chunk index)
  * quality: quality_metrics_complete.evaluate over 197 development and 3,984
    held-out Tagalog questions (held-out is verification only)

Tuklas pruned runs through both paths as the harness anchor; its known scores
(dev 0.9441624365482234, held-out 0.9357429718875502) must reproduce.

google/embeddinggemma-300m is gated.  The coordinator reads a Hugging Face
token from Secrets Manager (AmazonSageMaker-hf-token) and passes it only to the
download step; it is never printed or written.  Every later step runs offline.
Each result is uploaded before any gate is judged.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/runtime-comparison")          # the v2 runtime child's fixed WORK
HF_CACHE = WORK / "hf-cache"
sys.path.insert(0, str(INPUT))

TUKLAS_DEVELOPMENT = 0.9441624365482234
TUKLAS_HELDOUT = 0.9357429718875502


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def quality_child(name):
    import numpy as np
    import torch
    from multiwikiqa_evaluation import normalized
    from quality_metrics_complete import evaluate

    plan = json.loads((WORK / "plan.json").read_text())
    spec = plan["systems"][name]
    corpus = json.loads((WORK / "corpus.json").read_text())
    cohorts = {k: json.loads((WORK / (k + ".json")).read_text()) for k in ("development", "heldout")}
    documents = [r["text"] for r in corpus["records"]]
    if spec["runtime"] == "static-cpu":
        from model2vec import StaticModel
        folder = WORK / "models" / name
        model = StaticModel.from_pretrained(str(folder))
        prefix = json.loads((folder / "config.json").read_text()).get("query_prefix", "")

        def encode(values, query=False):
            return model.encode([prefix + v for v in values] if query else values,
                                batch_size=1024, max_length=None, use_multiprocessing=False)
    else:
        from aws_runtime_comparison_v2 import load_sentence_transformer
        torch.set_num_threads(int(plan["protocol"]["cpu_threads"]))
        model = load_sentence_transformer(spec["model"], "cuda")
        query_prefix, document_prefix = spec["model"]["query_prefix"], spec["model"]["document_prefix"]
        texts = [document_prefix + d for d in documents] + \
                [query_prefix + q["question"] for c in cohorts.values() for q in c]
        lengths = [len(x) for x in model.tokenizer(texts, truncation=False, padding=False)["input_ids"]]
        if max(lengths) > model.max_seq_length:
            raise ValueError("an input exceeds the model context; refusing silent truncation")

        def encode(values, query=False):
            return model.encode(values, batch_size=int(spec["model"]["batch_size"]),
                                prompt=query_prefix if query else document_prefix,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
    D = normalized(np.asarray(encode(documents), dtype=np.float32))
    result = {"system": name, "dimension": int(D.shape[1])}
    for cohort_name, cohort in cohorts.items():
        Q = normalized(np.asarray(encode([q["question"] for q in cohort], query=True), dtype=np.float32))
        summary, rows = evaluate(D, Q, cohort, corpus["owners"], corpus["parents"], corpus["records"])
        path = WORK / "quality" / f"{name}-{cohort_name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        result[cohort_name] = {**summary, "rankings_sha256": digest(path)}
    with (WORK / "quality" / f"{name}.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)


def coordinator():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
         "model2vec==0.9.0", "sentence-transformers==5.7.0", "transformers==5.3.0",
         "huggingface-hub==1.33.0", "numpy==2.5.3", "boto3==1.43.108", "psutil==7.1.3"],
        check=True)
    import boto3
    from aws_runtime_comparison_v2 import build_index, multipart_etag
    from multiwikiqa_ranking_v5 import prepare

    plan = json.loads((INPUT / "plan.json").read_text())
    for filename, expected in plan["staged_sha256"].items():
        if digest(INPUT / filename) != expected:
            raise ValueError("staged input changed: " + filename)
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "models").mkdir()
    HF_CACHE.mkdir()
    bucket, output = plan["bucket"], plan["output_prefix"].rstrip("/") + "/"
    s3 = boto3.client("s3", region_name=plan["region"])

    def publish(path, relative):
        s3.upload_file(str(path), bucket, output + relative)

    def event(stage, **details):
        print(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **details}), flush=True)

    layout = json.loads((INPUT / "shared-layout.json").read_text())
    records, owners = build_index(layout)
    annotations = [json.loads(line) for line in
                   (INPUT / "annotations.all.jsonl").read_text().splitlines() if line.strip()]
    tagalog = [q for q in annotations if q["language"] == "tl"]
    _, development, _ = prepare(tagalog, layout["chunks"])
    heldout = [q for q in tagalog if q["partition"] == "evaluation"]
    if len(development) != 197 or len(heldout) != 3_984 or \
            {q["query_id"] for q in development} & {q["query_id"] for q in heldout}:
        raise ValueError("cohort mismatch")
    (WORK / "corpus.json").write_text(json.dumps({
        "records": records, "owners": owners, "parents": layout["parents"],
        "sample_indices": list(range(0, len(records), 20))}))
    (WORK / "development.json").write_text(json.dumps(development))
    (WORK / "heldout.json").write_text(json.dumps(heldout))     # read only by the quality child
    event("index-built", chunks=len(records))

    # ---- stage Tuklas (S3, ETag-verified) ------------------------------------
    tuklas = plan["systems"]["tuklas-pruned"]
    target = WORK / "models" / "tuklas-pruned"
    target.mkdir()
    prefix = tuklas["s3_prefix"]
    for row in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        relative = row["Key"][len(prefix):]
        if relative:
            s3.download_file(bucket, row["Key"], str(target / relative))
    if multipart_etag(target / "model.safetensors", tuklas["etag_part_bytes"]) != tuklas["weights_etag"]:
        raise ValueError("Tuklas package differs from the S3 object")

    # ---- download EmbeddingGemma with the token; nothing after this needs it -
    gemma = plan["systems"]["embeddinggemma-gpu"]["model"]
    # Option B: the launcher may pass HF_TOKEN as a job environment variable.
    # Remove it from this process so no child except the download inherits it.
    token = os.environ.pop("HF_TOKEN", "").strip() or \
        boto3.client("secretsmanager", region_name=plan["region"]).get_secret_value(
            SecretId=plan["hf_token_secret"])["SecretString"].strip()
    environment = dict(os.environ)
    threads = str(plan["protocol"]["cpu_threads"])
    environment.update({"PYTHONPATH": os.pathsep.join(filter(None, [str(INPUT), os.environ.get("PYTHONPATH")])),
                        "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                        "OPENBLAS_NUM_THREADS": threads, "HF_HOME": str(HF_CACHE)})
    runtime_script = str(INPUT / "aws_runtime_comparison_v2.py")
    subprocess.run([sys.executable, runtime_script, "--prefetch", gemma["id"], gemma["revision"]],
                   env=environment | {"HF_TOKEN": token}, check=True, timeout=1800)
    del token
    snapshot = HF_CACHE / ("models--" + gemma["id"].replace("/", "--")) / "snapshots" / gemma["revision"]
    if not snapshot.is_dir():
        raise FileNotFoundError("prefetch left no pinned snapshot")
    prompts = json.loads((snapshot / "config_sentence_transformers.json").read_text()).get("prompts", {})
    if not prompts.get("query") or not prompts.get("document"):
        raise ValueError(f"model ships no query/document prompts: {sorted(prompts)}")
    gemma["query_prefix"], gemma["document_prefix"] = prompts["query"], prompts["document"]
    (WORK / "plan.json").write_text(json.dumps(plan))
    (WORK / "snapshots.json").write_text(json.dumps({gemma["id"]: str(snapshot)}))
    event("staged", prompts=prompts)

    offline = environment | {"HF_HUB_OFFLINE": "1"}
    results, failures = {"runtime": {}, "quality": {}}, {}
    try:
        for name in plan["order"]:
            for kind, script, flag, folder in (
                    ("runtime", runtime_script, "--child", WORK / "results"),
                    ("quality", str(INPUT / "aws_embeddinggemma_eval.py"), "--quality", WORK / "quality")):
                event(kind + "-start", system=name)
                try:
                    code = subprocess.run([sys.executable, script, flag, name], env=offline,
                                          timeout=plan["systems"][name]["timeout_seconds"]).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
                path = folder / (name + ".json")
                if code != 0 or not path.exists():
                    failures[f"{name}/{kind}"] = f"exited {code}"
                    event(kind + "-failed", system=name, code=code)
                    continue
                publish(path, f"{kind}/{name}.json")                 # preserve before judging
                if kind == "quality":
                    for cohort in ("development", "heldout"):
                        publish(folder / f"{name}-{cohort}.jsonl", f"quality/rankings/{name}-{cohort}.jsonl")
                results[kind][name] = json.loads(path.read_text())
                event(kind + "-complete", system=name)
    finally:
        gates = {}
        anchor = results["quality"].get("tuklas-pruned")
        if anchor:
            gates["tuklas-quality"] = {
                "development": anchor["development"]["Recall@10"], "heldout": anchor["heldout"]["Recall@10"],
                "passed": abs(anchor["development"]["Recall@10"] - TUKLAS_DEVELOPMENT) < 1e-9
                and abs(anchor["heldout"]["Recall@10"] - TUKLAS_HELDOUT) < 1e-9}
        for kind in ("runtime",):
            for name, result in results[kind].items():
                if result.get("development_gate"):
                    gates[f"{name}-{kind}-dev-vs-quality"] = {
                        "runtime_dev": result["development_gate"]["development_recall_at_10"],
                        "quality_dev": (results["quality"].get(name) or {}).get("development", {}).get("Recall@10")}
        report = {"status": "complete" if not failures and gates.get("tuklas-quality", {}).get("passed") else "needs-review",
                  "results": results, "failures": failures, "gates": gates, "prompts": prompts,
                  "heldout_use": "verification only; nothing is selected or tuned on it",
                  "limitations": plan["limitations"]}
        path = WORK / "embeddinggemma-report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        publish(path, "embeddinggemma-report.json")
    event("complete", status=report["status"], failures=failures)
    if report["status"] != "complete":
        raise SystemExit("results preserved, but a step failed or the anchor did not reproduce")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--quality":
        quality_child(sys.argv[2])
    elif len(sys.argv) == 1:
        coordinator()
    else:
        raise SystemExit("usage: aws_embeddinggemma_eval.py [--quality SYSTEM]")
