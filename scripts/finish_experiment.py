"""Finish all remaining measurements and the frozen held-out run after TokenLearn."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from traceback import format_exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.experiment import final_run_preflight
from pycon_model2vec_demo.model_registry import MODEL_NAMES, local_model_artifacts


STATUS = ROOT / "runs/completion-status.json"
TOKENLEARN_STATUS = ROOT / "runs/tokenlearn-pipeline-status.json"


def report(stage: str, state: str, detail: str = "") -> None:
    with STATUS.open("w", encoding="utf-8") as target:
        json.dump({"stage": stage, "state": state, "detail": detail}, target, indent=2)
        target.write("\n")


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / arguments[0]), *arguments[1:]], cwd=ROOT, check=True)


def main() -> None:
    try:
        report("tokenlearn", "waiting", "Waiting for the existing TokenLearn pipeline; no duplicate training process is started.")
        while True:
            state = json.loads(TOKENLEARN_STATUS.read_text(encoding="utf-8"))
            if state["state"] == "complete":
                break
            if state["state"] == "failed":
                raise RuntimeError(f"TokenLearn pipeline failed: {state['detail']}")
            time.sleep(30)

        artifacts = local_model_artifacts(ROOT, require_all=True)
        for name in MODEL_NAMES:
            artifact = artifacts[name]
            target = ROOT / "runs" / f"measurement-{name}-{artifact.fingerprint[:12]}.json"
            if not target.exists():
                report("measure", "running", name)
                run("measure_model.py", "--model", name)
            # If the normal run reused a development cache, capture a separate
            # full-corpus embedding-build time without overwriting the baseline.
            measurement = json.loads(target.read_text(encoding="utf-8"))
            if measurement["full_corpus_embedding_build_seconds"] is None:
                build_target = ROOT / "runs" / f"measurement-build-{name}-{artifact.fingerprint[:12]}.json"
                if not build_target.exists():
                    report("measure-fresh-corpus-build", "running", name)
                    run("measure_model.py", "--model", name, "--fresh-embedding-build")

        final_run_preflight(ROOT)
        existing = list((ROOT / "runs").glob("heldout-five-model-*.json"))
        if existing:
            raise FileExistsError("Held-out report already exists; refusing another final evaluation")
        report("heldout-five-model", "running", "All artifact and measurement preflights passed.")
        run("run_quality.py", "--split", "heldout")
        reports = list((ROOT / "runs").glob("heldout-five-model-*.json"))
        if len(reports) != 1 or not reports[0].with_suffix(".md").is_file():
            raise RuntimeError("Final report pair was not created")
        report("complete", "complete", str(reports[0].relative_to(ROOT)))
    except Exception:
        report("failed", "failed", format_exc())
        raise


if __name__ == "__main__":
    main()
