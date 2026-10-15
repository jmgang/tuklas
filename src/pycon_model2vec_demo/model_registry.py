"""Local-only artifact registry for the frozen five-model experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .distillation import file_hashes
from .encoders import RetrievalEncoder, load_e5_encoder, load_static_encoder


MODEL_NAMES = (
    "teacher", "potion_retrieval", "plain_student", "custom_student", "tokenlearn_custom_student",
)


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    path: Path
    settings: dict
    files_sha256: dict[str, str]
    fingerprint: str

    def load_encoder(self) -> RetrievalEncoder:
        if self.name == "teacher":
            return load_e5_encoder(
                str(self.path), max_length=self.settings["max_length"], batch_size=self.settings["batch_size"],
            )
        return load_static_encoder(
            str(self.path), max_length=self.settings["max_length"], batch_size=self.settings["batch_size"],
            query_prefix=self.settings["query_prefix"], passage_prefix=self.settings["passage_prefix"],
        )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_model_artifacts(project_dir: Path, *, require_all: bool = True) -> dict[str, ModelArtifact]:
    """Resolve and verify all frozen local models; never download substitutes."""
    models = _json(project_dir / "configs/models-v1.json")
    tokenlearn = _json(project_dir / "configs/tokenlearn-v1.json")
    suffix = models["teacher"]["revision"][:12]
    plain_record = _json(project_dir / "runs" / f"plain-e5-{suffix}-provenance.json")
    custom_record = _json(project_dir / "runs" / f"custom-e5-{suffix}-provenance.json")
    development_record = _json(project_dir / "runs/development-check-v2.json")
    entries = {
        "teacher": (
            project_dir / "models" / f"e5-base-v2-{suffix}",
            models["teacher"], plain_record["teacher"]["files_sha256"],
        ),
        "potion_retrieval": (
            project_dir / "models" / f"potion-retrieval-{models['ready_made_baseline']['revision'][:12]}",
            models["ready_made_baseline"], development_record["models"]["potion_retrieval"]["artifact_files_sha256"],
        ),
        "plain_student": (
            project_dir / "models" / f"plain-e5-{suffix}",
            models["students"]["plain"], plain_record["student"]["files_sha256"],
        ),
        "custom_student": (
            project_dir / "models" / f"custom-e5-{suffix}",
            models["students"]["plain"], custom_record["student"]["files_sha256"],
        ),
    }
    tokenlearn_path = project_dir / tokenlearn["output_model"]
    tokenlearn_record_path = project_dir / "runs" / f"tokenlearn-custom-e5-{suffix}-provenance.json"
    if tokenlearn_record_path.exists() and tokenlearn_path.exists():
        tokenlearn_record = _json(tokenlearn_record_path)
        if tokenlearn_record["custom_student_files_sha256"] != custom_record["student"]["files_sha256"]:
            raise ValueError("TokenLearn model was not trained from this custom student")
        entries["tokenlearn_custom_student"] = (
            tokenlearn_path, models["students"]["plain"], tokenlearn_record["student"]["files_sha256"],
        )
    elif require_all:
        raise FileNotFoundError("The TokenLearn-trained fifth model is not complete; held-out evaluation is closed")
    resolved = {}
    for name, (path, settings, expected) in entries.items():
        if not path.is_dir():
            raise FileNotFoundError(path)
        actual = file_hashes(path)
        if actual != expected:
            raise ValueError(f"{name} files differ from frozen provenance")
        fingerprint = sha256(json.dumps({"files": actual, "settings": settings}, sort_keys=True).encode()).hexdigest()
        resolved[name] = ModelArtifact(name, path, settings, actual, fingerprint)
    return resolved
