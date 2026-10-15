"""Fail-fast checks for the actual pinned TokenLearn cloud environment."""
import importlib
import importlib.metadata
import json

from packaging.requirements import Requirement


def verify_runtime(expected_commit):
    """Check base requirements and import real entrypoints before teacher work."""
    checked = {}
    for text in importlib.metadata.requires("tokenlearn") or []:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        version = importlib.metadata.version(requirement.name)
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            raise RuntimeError(f"Unsatisfied TokenLearn dependency: {requirement}, installed {version}")
        checked[requirement.name] = version
    direct_url = importlib.metadata.distribution("tokenlearn").read_text("direct_url.json")
    if not direct_url or json.loads(direct_url).get("vcs_info", {}).get("commit_id") != expected_commit:
        raise RuntimeError("Installed TokenLearn Git commit differs from frozen source")
    modules = ("tokenlearn.featurize", "tokenlearn.train", "model2vec.train", "sealion_training", "sealion_source_integrity")
    for name in modules:
        importlib.import_module(name)
    return {"requirements": checked, "imported_entrypoints": list(modules),
            "tokenlearn_commit": expected_commit}

