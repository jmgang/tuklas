"""Resume full-corpus TokenLearn features, train the student, and compare it."""

import json
from pathlib import Path
import subprocess
import sys
from traceback import format_exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.tokenlearn_training import featurize_corpus, train_custom_student


STATUS = ROOT / "runs/tokenlearn-pipeline-status.json"


def report(stage: str, state: str, detail: str = "") -> None:
    with STATUS.open("w", encoding="utf-8") as target:
        json.dump({"stage": stage, "state": state, "detail": detail}, target, indent=2)
        target.write("\n")


def main() -> None:
    config = ROOT / "configs/tokenlearn-v1.json"
    features_manifest = ROOT / "runs/tokenlearn-features-v1/manifest.json"
    model_record = ROOT / "runs/tokenlearn-custom-e5-f52bf8ec8c71-provenance.json"
    try:
        if not features_manifest.exists():
            report("featurize", "running")
            featurize_corpus(ROOT, config)
        if not model_record.exists():
            report("train", "running")
            train_custom_student(ROOT, config)
        if not (ROOT / "runs/development-check-v3.json").exists():
            report("development-preflight", "running")
            subprocess.run([sys.executable, str(ROOT / "scripts/development_check.py")], cwd=ROOT, check=True)
        if not list((ROOT / "runs").glob("tokenlearn-development-comparison-*.json")):
            report("development-comparison", "running")
            subprocess.run([sys.executable, str(ROOT / "scripts/compare_custom_students.py")], cwd=ROOT, check=True)
        report("complete", "complete")
    except Exception:
        report("failed", "failed", format_exc())
        raise


if __name__ == "__main__":
    main()
