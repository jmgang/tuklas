"""Authorized local DisastIR quality repair; no AWS, HTTP, latency or cleanup.

Official Sentence Transformers ranking losses adapt the frozen TokenLearn
student. Only original development queries may train/select a checkpoint.
Evaluation corpus/cohort/golds remain unchanged. Outputs are new immutable
files; failed or superseded outputs are never removed or overwritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from model2vec import StaticModel
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.grade3 import Grade3Query, TOP10_METRICS, score_top10_query, summarize, select_cohorts
from pycon_model2vec_demo.parent_retrieval import parent_top_k
from pycon_model2vec_demo.retrieval import _best_indices
from pycon_model2vec_demo.retrieval_finetuning import independent_state, safe_batches, split_development


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def event(directory, stage, **values):
    value = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
    save(directory / "events" / f"{uuid4().hex}.json", value)
    print(json.dumps(value), flush=True)


class ExportParityEmbedding(StaticEmbedding):
    """Use official trainable embedding module with Model2Vec UNK filtering."""
    def __init__(self, base):
        tokenizer = Tokenizer.from_str(base.tokenizer.to_str())
        tokenizer.no_truncation()
        super().__init__(tokenizer, embedding_weights=torch.from_numpy(base.embedding.astype(np.float32).copy()))
        self.unknown = base.unk_token_id

    def preprocess(self, inputs, prompt=None, **kwargs):
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        rows = self.tokenizer.encode_batch(inputs, add_special_tokens=False)
        ids = [[i for i in row.ids if i != self.unknown] for row in rows]
        return {"input_ids": torch.tensor([i for row in ids for i in row], dtype=torch.long),
                "offsets": torch.tensor(np.cumsum([0] + [len(row) for row in ids[:-1]]), dtype=torch.long)}


def export(module, base):
    return StaticModel(vectors=module.embedding.weight.detach().cpu().numpy().copy(),
                       tokenizer=Tokenizer.from_str(module.tokenizer.to_str()),
                       config={**base.config, "normalize": True}, normalize=True)


def check_parity(model, static):
    strings = ["query: How do shelters work?", "passage: A shelter opens after flooding.",
               "query: What about 🛸 and 日本語?", "passage: oysters acidification"]
    with torch.no_grad():
        trained = torch.nn.functional.normalize(model(model.tokenize(strings))["sentence_embedding"], dim=1).numpy()
    served = static.encode(strings, max_length=None, use_multiprocessing=False)
    if not np.allclose(trained, served, atol=2e-6):
        raise ValueError("Training/export encoding parity failed")
    return float(np.max(np.abs(trained - served)))


def vector_matrix(model, texts, prefix, directory):
    directory.mkdir(parents=True, exist_ok=False)
    chunks = []
    for start in range(0, len(texts), 8192):
        rows = np.asarray(model.encode([prefix+t for t in texts[start:start+8192]],
                          max_length=None, batch_size=1024, use_multiprocessing=False), dtype=np.float32)
        if rows.shape != (min(8192, len(texts)-start), model.dim) or not np.isfinite(rows).all():
            raise ValueError("Invalid segment vectors")
        path = directory / f"{start:06d}.npy"
        with path.open("xb") as stream:
            np.save(stream, rows, allow_pickle=False)
        save(path.with_suffix(".json"), {"start": start, "stop": start+len(rows), "sha256": file_hash(path)})
        chunks.append(rows)
        if len(chunks) % 8 == 0 or start+len(rows) == len(texts):
            event(directory.parent, "segment-encoding", vector_directory=directory.name,
                  rows=start+len(rows), total=len(texts))
    return np.concatenate(chunks)


def score(model, vectors, queries, prefix, owners, corpus, directory, stage):
    rows = []
    for start in range(0, len(queries), 32):
        batch = queries[start:start+32]
        qv = model.encode([prefix+q.query for q in batch], max_length=None,
                          use_multiprocessing=False, batch_size=1024)
        rankings = parent_top_k(qv, vectors, owners, corpus.ids, k=10, block_size=8192)
        for query, ranking in zip(batch, rankings):
            rows.append({"query_id": query.query_id, "category": query.category,
                         "metrics": score_top10_query(query, ranking),
                         "top10": [{"corpus_id": r.corpus_id, "similarity": r.similarity} for r in ranking]})
        event(directory, stage, queries=len(rows), total=len(queries))
    return {"per_query": rows, "summary": summarize(rows, metrics=TOP10_METRICS)}


def oyster_ranks(model, vectors, query, prefix, owners, corpus):
    q = model.encode([prefix+query.query], max_length=None, use_multiprocessing=False)[0]
    scores = np.full(len(corpus.ids), -np.inf, dtype=np.float32)
    for start in range(0, len(owners), 8192):
        np.maximum.at(scores, owners[start:start+8192], vectors[start:start+8192] @ q)
    index = {key: i for i, key in enumerate(corpus.ids)}
    ranks = []
    for key in sorted(query.gold_ids):
        i = index[key]
        rank = 1 + int((scores > scores[i]).sum())
        rank += sum(corpus.ids[j] < key for j in np.flatnonzero(scores == scores[i]))
        ranks.append({"corpus_id": key, "rank": rank, "score": float(scores[i]), "text": corpus.texts[i]})
    return {"query_id": query.query_id, "query": query.query, "gold_ranks": ranks,
            "recall_at_10": sum(r["rank"] <= 10 for r in ranks)/len(ranks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Run id must be one folder name")
    directory = ROOT / "runs" / args.run_id
    directory.mkdir(exist_ok=False)
    config_path = ROOT / "configs/disastir-retrieval-repair-v1.json"
    config = read(config_path)
    save(directory / "settings.json", config)
    event(directory, "preflight", latency_measured=False, aws_used=False)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    data = read(ROOT / "configs/disastir-qa-all-categories.json")
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=config["corpus_sha256"])
    prepared = load_qa_collection(data_dir=ROOT/"data/raw", revision=data["revision"],
        category_checksums=data["categories"], seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"])
    mapped_test, _ = map_query_groups(prepared.evaluation, corpus)
    _, test_queries, _ = select_cohorts(prepared.evaluation, mapped_test, corpus)
    catalog = read(ROOT / config["verified_reference"] / "shared-catalog.json")
    if [q.query_id for q in test_queries] != catalog["query_ids"]:
        raise ValueError("Evaluation cohort changed")
    if len(corpus.ids) != 239704 or len(catalog["segments"]) != 240765 or len(test_queries) != 1127:
        raise ValueError("Evaluation size changed")
    segments = tuple(corpus.texts[i][a:b] for i,a,b in catalog["segments"])
    owners = np.array([i for i,_,_ in catalog["segments"]], dtype=np.int64)
    rows = {key:i for i,key in enumerate(corpus.ids)}
    parent_segment_count = np.bincount(owners, minlength=len(corpus.ids))
    mapped_dev, _ = map_query_groups(prepared.development, corpus)
    _, development_queries, _ = select_cohorts(prepared.development, mapped_dev, corpus)
    forbidden = {q.query_id for q in prepared.evaluation}
    train, validation = split_development(development_queries, forbidden,
        seed=config["split_seed"], validation_fraction=config["validation_fraction"])
    mapped_by_id = {q.query_id:q for q in mapped_dev}
    eligible = []
    for query in train:
        mapped = mapped_by_id[query.query_id]
        positive = sorted(key for key in query.gold_ids if parent_segment_count[rows[key]] == 1)
        negative = sorted(key for key,grade in mapped.judgments.items()
                          if grade == 0 and parent_segment_count[rows[key]] == 1)
        if positive and negative:
            eligible.append({"query": query, "positives": positive, "negatives": negative,
                             "positive_ids": {key for key,grade in mapped.judgments.items() if grade > 0}})
    if not eligible:
        raise ValueError("No eligible development training examples")
    reference = read(ROOT / config["verified_reference"] / "reports/combined-report.json")
    for name in ("tokenlearn_custom_student", "potion_retrieval"):
        frozen = reference["protocol"]["frozen_models"][name]
        for filename, checksum in frozen["files_sha256"].items():
            if file_hash(ROOT/frozen["path"]/filename) != checksum:
                raise ValueError(f"Changed model artifact {name}: {filename}")
    manifest = {"training_query_ids": [x["query"].query_id for x in eligible],
                "validation_query_ids": [q.query_id for q in validation],
                "evaluation_query_ids": [q.query_id for q in test_queries],
                "source_development_queries": len(prepared.development),
                "grade3_development_queries": len(development_queries),
                "training_queries": len(eligible), "validation_queries": len(validation),
                "ineligible_training_queries": len(train)-len(eligible),
                "corpus_sha256": config["corpus_sha256"], "shared_catalog_sha256": file_hash(ROOT/config["verified_reference"]/"shared-catalog.json"),
                "source_sha256": file_hash(Path(__file__)), "config_sha256": file_hash(config_path),
                "protocol": config["protocol_note"]}
    assert not set(manifest["training_query_ids"]) & forbidden
    save(directory/"manifest.json", manifest)
    event(directory, "training-data-frozen", train=len(eligible), validation=len(validation), evaluation=len(test_queries))
    base = StaticModel.from_pretrained(str(ROOT/config["initial_student"]))
    module = ExportParityEmbedding(base)
    model = SentenceTransformer(modules=[module], device="cpu")
    parity = check_parity(model, base)
    event(directory, "export-parity-passed", max_absolute_difference=parity)
    criterion = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), config["matryoshka_dimensions"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_key, best_epoch, best_state = None, None, None
    validation_history = []
    def validate(epoch):
        nonlocal best_key, best_epoch, best_state
        model.eval()
        static = export(module, base)
        vectors = vector_matrix(static, segments, "passage: ", directory/f"validation-vectors-epoch-{epoch:02d}")
        report = score(static, vectors, validation, "query: ", owners, corpus, directory, f"validation-epoch-{epoch}")
        save(directory/f"validation-epoch-{epoch:02d}.json", report)
        metrics = report["summary"]["overall"]
        key = (metrics["recall_at_10"], metrics["ndcg_at_10"])
        validation_history.append({"epoch":epoch, **metrics})
        if best_key is None or key > best_key:
            best_key, best_epoch, best_state = key, epoch, independent_state(model)
        with (directory/f"weights-epoch-{epoch:02d}.npy").open("xb") as stream:
            np.save(stream, static.embedding, allow_pickle=False)
        event(directory,"checkpoint-validated",epoch=epoch,recall_at_10=key[0],ndcg_at_10=key[1],best_epoch=best_epoch)
        del vectors
        gc.collect()
    validate(0)
    generator = random.Random(config["seed"])
    training_schedule = []
    for epoch in range(1,config["epochs"]+1):
        training_rows = []
        for item in eligible:
            query = item["query"]
            training_rows.append({"query_id": query.query_id, "query":query.query,
                "positive_id":generator.choice(item["positives"]), "negative_id":generator.choice(item["negatives"]),
                "positive_ids":item["positive_ids"]})
        generator.shuffle(training_rows)
        training_schedule.append(list(safe_batches(training_rows,config["batch_size"])))
    total_steps = sum(len(batches) for batches in training_schedule)
    warmup_steps = max(1, round(total_steps * config["warmup_fraction"]))
    step = 0
    for epoch, batches in enumerate(training_schedule, 1):
        model.train()
        losses=[]
        used=0
        for batch in batches:
            scale = (step+1)/warmup_steps if step < warmup_steps else (total_steps-step)/max(1,total_steps-warmup_steps)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = config["learning_rate"] * scale
            optimizer.zero_grad(set_to_none=True)
            features = [model.tokenize(texts) for texts in (
                ["query: "+r["query"] for r in batch],
                ["passage: "+corpus.texts[rows[r["positive_id"]]] for r in batch],
                ["passage: "+corpus.texts[rows[r["negative_id"]]] for r in batch])]
            loss = criterion(features, torch.empty(0))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ranking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
            used += len(batch)
        save(directory/f"training-epoch-{epoch:02d}.json",{"epoch":epoch,"loss_mean":float(np.mean(losses)),"batches":len(losses),"used_queries":used})
        event(directory,"ranking-training",epoch=epoch,epochs=config["epochs"],loss=float(np.mean(losses)),used_queries=used)
        if epoch in config["checkpoint_evaluation_epochs"]:
            validate(epoch)
    model.load_state_dict(best_state)
    model.eval()
    selected = export(module,base)
    parity = check_parity(model,selected)
    output = directory/"model"
    output.mkdir(exist_ok=False)
    selected.save_pretrained(output)
    model_hashes = {str(p.relative_to(output)):file_hash(p) for p in output.rglob("*") if p.is_file()}
    save(directory/"selection.json",{"epoch":best_epoch,"validation_history":validation_history,"parity_max_absolute_difference":parity,"model_files_sha256":model_hashes,
         "objective":"official MNRL + MatryoshkaLoss; original TokenLearn-pretrained initialization", "best_state_cloned":True})
    event(directory,"model-selected",epoch=best_epoch,dimension=selected.dim)
    oyster = next(q for q in test_queries if q.query_id=="0291b8ce4f12580d907808480c1d1f64c3efb47d2a9fa9daec64ad3e85a04865")
    reports={}
    for name,static,prefix in (("potion_retrieval",StaticModel.from_pretrained(str(ROOT/config["potion_baseline"])),""),
                               ("tokenlearn_retrieval_student",selected,"passage: ")):
        vectors=vector_matrix(static,segments,prefix,directory/f"evaluation-vectors-{name}")
        report=score(static,vectors,test_queries,"" if name=="potion_retrieval" else "query: ",owners,corpus,directory,f"evaluation-{name}")
        report["model"] = name
        report["provenance"] = manifest
        save(directory/f"quality-{name}.json",report)
        save(directory/f"oyster-{name}.json",oyster_ranks(static,vectors,oyster,"" if name=="potion_retrieval" else "query: ",owners,corpus))
        reports[name]=report["summary"]
        del vectors
        gc.collect()
    save(directory/"comparison.json",{"models":reports,"selected_epoch":best_epoch,"provenance":manifest,"latency_measured":False,
         "limitations":["Post-hoc repair after prior evaluation inspection; not a newly blind test.",
                        "New labeled adaptation uses development data only; test queries and qrels never train or select checkpoints.",
                        "Training uses official Potion loss family but not its full training mixture, capacity, batch size or budget.",
                        "Pooled incomplete DisastIR relevance judgments remain unchanged."]})
    event(directory,"complete",results={name:report["overall"] for name,report in reports.items()})


if __name__ == "__main__":
    main()
