"""CPU tokenizer/data preflight on ALL training text, before paid replacement."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pycon_model2vec_demo.sealion_training import safe_windows
from pycon_model2vec_demo.sealion_source_integrity import validate_source_text


def main():
    from transformers import AutoTokenizer
    config = json.loads((ROOT / "configs/sealion-two-students-v1.json").read_text())
    directory = ROOT / "runs/sealion-local-preflight-20261004-v3"
    directory.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(config["teacher"]["model_id"],
        revision=config["teacher"]["revision"], trust_remote_code=False,
        cache_dir=str(directory / "teacher-tokenizer"))
    result = {"teacher": config["teacher"], "arms": {}, "all_training_text_preflighted": False,
              "original_text_unchanged": True, "no_qa_inputs": True}
    for name, arm in config["arms"].items():
        source_count = 0
        window_count = 0
        language_counts = Counter()
        for filename, spec in arm["files"].items():
            path = ROOT / spec["path"]
            value = sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024*1024), b""):
                    value.update(block)
            if value.hexdigest() != spec["sha256"]:
                raise ValueError("Whole-file byte checksum changed: " + filename)
            with path.open() as source:
                count = 0
                for line in source:
                    row = json.loads(line)
                    count += 1
                    if filename == "parents.jsonl":
                        if not row["component_id"] or not row["parent_id"]:
                            raise ValueError("Invalid parent identity")
                        continue
                    text = validate_source_text(row, name)
                    if row["language"] not in ("en", "tl"):
                        raise ValueError("Unexpected language")
                    source_count += 1
                    windows = safe_windows(text, tokenizer)
                    window_count += len(windows)
                    language_counts[row["language"]] += 1
                    if source_count % 10000 == 0:
                        print(json.dumps({"arm": name, "source_rows_checked": source_count}), flush=True)
                if count != spec["rows"]:
                    raise ValueError("Source row count changed: " + filename)
        result["arms"][name] = {"source_rows": source_count, "safe_windows": window_count,
                               "source_language_counts": dict(language_counts)}
        print(json.dumps(result["arms"][name]), flush=True)
    result["all_training_text_preflighted"] = True
    with (directory / "receipt.json").open("x") as target:
        json.dump(result, target, sort_keys=True, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
