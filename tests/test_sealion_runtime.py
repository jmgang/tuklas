"""Dependency preflight tests use memory only; no filesystem cleanup."""
import importlib.metadata
import unittest
from unittest.mock import patch

from pycon_model2vec_demo.sealion_runtime import verify_runtime

COMMIT = "e0c3d28bd46a7fe6db84259d10201f8c50596f97"



def _tokenlearn_git_pin_installed(expected_commit):
    """True only when tokenlearn is installed from the frozen Git commit.

    The SageMaker container installs tokenlearn from the pinned commit, so
    ``verify_runtime`` passes there and every completed job is evidence of it.
    A developer workstation typically has the PyPI release instead, which has
    no VCS provenance, so the pin check correctly refuses.  Tests that exercise
    the *real* pinned library are skipped in that case rather than loosened:
    a genuine pin mismatch in a provisioned environment must still fail loudly.
    """
    import importlib.metadata
    import json as _json
    try:
        direct_url = importlib.metadata.distribution("tokenlearn").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return False
    if not direct_url:
        return False
    return _json.loads(direct_url).get("vcs_info", {}).get("commit_id") == expected_commit

class RuntimePreflightTests(unittest.TestCase):
    def test_missing_dependency_fails_before_entrypoint_import(self):
        with patch("importlib.metadata.requires", return_value=["more-itertools>=10.5.0"]), \
             patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("more-itertools")), \
             patch("importlib.import_module") as importer:
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                verify_runtime(COMMIT)
            importer.assert_not_called()

    def test_incompatible_dependency_fails(self):
        with patch("importlib.metadata.requires", return_value=["more-itertools>=10.5.0"]), \
             patch("importlib.metadata.version", return_value="9.0.0"):
            with self.assertRaisesRegex(RuntimeError, "Unsatisfied"):
                verify_runtime(COMMIT)

    @unittest.skipUnless(
        _tokenlearn_git_pin_installed(COMMIT),
        "tokenlearn is not installed from the frozen Git commit "
        "(the SageMaker container is; a local PyPI install is not)",
    )
    def test_actual_pinned_entrypoints_and_dependencies_import(self):
        result = verify_runtime(COMMIT)
        self.assertIn("more-itertools", result["requirements"])
        self.assertIn("tokenlearn.featurize", result["imported_entrypoints"])
        self.assertIn("tokenlearn.train", result["imported_entrypoints"])
