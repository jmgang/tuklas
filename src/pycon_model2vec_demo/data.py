"""Validated judged-candidate groups from a pinned, local DisastIR release."""

from dataclasses import dataclass
from hashlib import sha256
import gzip
import json
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


QA_CATEGORIES = frozenset(
    {
        "biological",
        "chemical",
        "environmental",
        "extraterrestrial",
        "geohazard",
        "meteorological_hydrological",
        "societal",
        "technological",
    }
)


@dataclass(frozen=True)
class Candidate:
    passage_id: str
    text: str
    relevance: int

    def __post_init__(self) -> None:
        if not self.passage_id or not self.text:
            raise ValueError("Candidate ID and text must be nonempty")
        if isinstance(self.relevance, bool) or not isinstance(self.relevance, int) or self.relevance < 0:
            raise ValueError("Relevance must be a nonnegative integer")


@dataclass(frozen=True)
class QueryGroup:
    query_id: str
    query: str
    candidates: tuple[Candidate, ...]
    category: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id or not self.query:
            raise ValueError("Query ID and text must be nonempty")
        if not self.candidates:
            raise ValueError("A query group needs at least one judged candidate")
        ids = [candidate.passage_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate IDs must be unique within a query group")


@dataclass(frozen=True)
class PreparedSlice:
    development: tuple[QueryGroup, ...]
    evaluation: tuple[QueryGroup, ...]
    manifest: dict


@dataclass(frozen=True)
class PreparedCollection:
    development: tuple[QueryGroup, ...]
    evaluation: tuple[QueryGroup, ...]
    manifest: dict


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"Cannot read source file {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json_array(path: Path) -> list:
    try:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8") as source:
            records = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON array from {path}: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def _required_text(record: object, field: str, location: str) -> str:
    if not isinstance(record, dict) or not isinstance(record.get(field), str):
        raise ValueError(f"{location}: missing or non-string {field}")
    value = record[field]
    if not value.strip():
        raise ValueError(f"{location}: empty {field}")
    return value


def source_path(data_dir: Path, filename: str) -> Path:
    """Find a pinned source as plain JSON or gzip, without fetching data."""
    plain = data_dir / filename
    compressed = data_dir / f"{filename}.gz"
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    raise ValueError(f"Missing source file: {plain} (or {compressed.name})")


def split_query_groups(
    groups: Sequence[QueryGroup], *, seed: str, evaluation_fraction: float
) -> tuple[list[QueryGroup], list[QueryGroup]]:
    """Split whole query groups deterministically; preserve input order within splits."""
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between 0 and 1")
    if len({group.query_id for group in groups}) != len(groups):
        raise ValueError("Query IDs must be unique before splitting")
    development: list[QueryGroup] = []
    evaluation: list[QueryGroup] = []
    for group in groups:
        # Repeated query wording stays in one split even if the release gives it two IDs.
        normalized_query = _normalize_query(group.query)
        digest = sha256(f"{seed}\0{normalized_query}".encode("utf-8")).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        (evaluation if fraction < evaluation_fraction else development).append(group)
    return development, evaluation


def load_disastir_slice(
    *,
    query_path: Path,
    qrels_path: Path,
    revision: str,
    task: str,
    category: str,
    seed: str,
    evaluation_fraction: float,
    expected_query_sha256: str | None = None,
    expected_qrels_sha256: str | None = None,
    orphan_qrels_policy: str = "error",
) -> PreparedSlice:
    """Load one published query/qrels pair; do not load the full corpus.

    Relevance labels are returned in both splits for later evaluation. The future
    distillation step must receive only permitted development/unlabeled text.
    """
    if task != "QA" or category not in QA_CATEGORIES:
        raise ValueError("Only DisastIR QA categories are supported")
    if not revision or not seed:
        raise ValueError("Dataset revision and split seed must be nonempty")
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between 0 and 1")
    if orphan_qrels_policy not in {"error", "report_exclude"}:
        raise ValueError("orphan_qrels_policy must be error or report_exclude")
    if query_path.name not in {f"{task}_{category}.json", f"{task}_{category}.json.gz"}:
        raise ValueError(f"Wrong query file for {task}/{category}: {query_path.name}")
    if qrels_path.name not in {
        f"{task}_{category}_qrels.json", f"{task}_{category}_qrels.json.gz"
    }:
        raise ValueError(f"Wrong qrels file for {task}/{category}: {qrels_path.name}")

    query_sha256 = _file_sha256(query_path)
    qrels_sha256 = _file_sha256(qrels_path)
    if expected_query_sha256 and query_sha256 != expected_query_sha256:
        raise ValueError("Query file SHA-256 differs from the pinned release")
    if expected_qrels_sha256 and qrels_sha256 != expected_qrels_sha256:
        raise ValueError("Qrels file SHA-256 differs from the pinned release")

    query_records = _read_json_array(query_path)
    qrel_records = _read_json_array(qrels_path)
    if not query_records or not qrel_records:
        raise ValueError("Query and qrels files must both contain records")

    query_order: list[str] = []
    seen_normalized: set[str] = set()
    for index, record in enumerate(query_records):
        query = _required_text(record, "user_query", f"query[{index}]")
        normalized = _normalize_query(query)
        if normalized in seen_normalized:
            raise ValueError(f"query[{index}]: duplicate normalized query text")
        seen_normalized.add(normalized)
        query_order.append(query)

    candidates_by_query: dict[str, list[Candidate]] = {query: [] for query in query_order}
    score_counts = {score: 0 for score in range(4)}
    orphan_counts: dict[str, int] = {}
    for index, record in enumerate(qrel_records):
        location = f"qrels[{index}]"
        query = _required_text(record, "user_query", location)
        passage = _required_text(record, "passage", location)
        if query not in candidates_by_query:
            if orphan_qrels_policy == "error":
                raise ValueError(f"{location}: query not found in query file")
            orphan_counts[query] = orphan_counts.get(query, 0) + 1
            continue
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or score not in score_counts:
            raise ValueError(f"{location}: score must be an integer from 0 to 3")
        passage_id = sha256(passage.encode("utf-8")).hexdigest()
        candidates_by_query[query].append(Candidate(passage_id, passage, score))
        score_counts[score] += 1

    groups: list[QueryGroup] = []
    for query in query_order:
        candidates = candidates_by_query[query]
        if not candidates:
            raise ValueError(f"Query has no judged candidates: {query[:80]}")
        query_id = sha256(_normalize_query(query).encode("utf-8")).hexdigest()
        groups.append(QueryGroup(query_id, query, tuple(candidates), category))

    development, evaluation = split_query_groups(
        groups, seed=seed, evaluation_fraction=evaluation_fraction
    )
    if not development or not evaluation:
        raise ValueError("Split produced an empty partition; use more queries or another seed")

    candidate_counts = [len(group.candidates) for group in groups]
    manifest = {
        "dataset": "DMIR01/DisastIR",
        "revision": revision,
        "task": task,
        "category": category,
        "query_file": query_path.name,
        "qrels_file": qrels_path.name,
        "query_sha256": query_sha256,
        "qrels_sha256": qrels_sha256,
        "score_counts": score_counts,
        "query_count": len(groups),
        "judged_pair_count": sum(score_counts.values()),
        "raw_judgment_count": len(qrel_records),
        "excluded_orphan_judgment_count": sum(orphan_counts.values()),
        "orphan_query_count": len(orphan_counts),
        "orphan_query_examples": [query[:120] for query in sorted(orphan_counts)[:3]],
        "candidate_count": {
            "min": min(candidate_counts),
            "median": median(candidate_counts),
            "max": max(candidate_counts),
        },
        "split_seed": seed,
        "evaluation_fraction": evaluation_fraction,
        "development_query_ids": [group.query_id for group in development],
        "evaluation_query_ids": [group.query_id for group in evaluation],
        "scope": "judged-candidate ranking; not full-corpus retrieval",
    }
    return PreparedSlice(tuple(development), tuple(evaluation), manifest)


def load_qa_collection(
    *, data_dir: Path, revision: str, category_checksums: Mapping[str, Mapping[str, str]],
    seed: str, evaluation_fraction: float
) -> PreparedCollection:
    """Load all configured QA categories, preserving category-level split counts."""
    if not category_checksums:
        raise ValueError("At least one QA category is required")
    development: list[QueryGroup] = []
    evaluation: list[QueryGroup] = []
    category_manifests: dict[str, dict] = {}
    for category in sorted(category_checksums):
        checksums = category_checksums[category]
        try:
            prepared = load_disastir_slice(
                query_path=source_path(data_dir, f"QA_{category}.json"),
                qrels_path=source_path(data_dir, f"QA_{category}_qrels.json"),
                revision=revision,
                task="QA",
                category=category,
                seed=seed,
                evaluation_fraction=evaluation_fraction,
                expected_query_sha256=checksums["query_sha256"],
                expected_qrels_sha256=checksums["qrels_sha256"],
                orphan_qrels_policy="report_exclude",
            )
        except ValueError as exc:
            raise ValueError(f"QA/{category}: {exc}") from exc
        category_manifests[category] = {
            **prepared.manifest,
            "development_count": len(prepared.development),
            "evaluation_count": len(prepared.evaluation),
        }
        development.extend(prepared.development)
        evaluation.extend(prepared.evaluation)
    development_ids = {group.query_id for group in development}
    evaluation_ids = {group.query_id for group in evaluation}
    if development_ids & evaluation_ids:
        raise ValueError("The same normalized query text appears across category splits")
    if len(development_ids) != len(development) or len(evaluation_ids) != len(evaluation):
        raise ValueError("Duplicate normalized query text appears across QA categories")
    manifest = {
        "dataset": "DMIR01/DisastIR",
        "revision": revision,
        "task": "QA",
        "categories": category_manifests,
        "query_count": len(development) + len(evaluation),
        "judged_pair_count": sum(item["judged_pair_count"] for item in category_manifests.values()),
        "raw_judgment_count": sum(item["raw_judgment_count"] for item in category_manifests.values()),
        "excluded_orphan_judgment_count": sum(
            item["excluded_orphan_judgment_count"] for item in category_manifests.values()
        ),
        "split_seed": seed,
        "evaluation_fraction": evaluation_fraction,
        "development_query_ids": [group.query_id for group in development],
        "evaluation_query_ids": [group.query_id for group in evaluation],
        "scope": "Data preparation only; the primary benchmark searches the full corpus",
    }
    return PreparedCollection(tuple(development), tuple(evaluation), manifest)
