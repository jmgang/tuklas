from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_worker_is_sealion_only_and_starts_with_official_model2vec_distill():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()

    assert '"aisingapore/SEA-LION-E5-Embedding-600M"' in source
    assert "potion-retrieval" not in source.lower()
    assert "multilingual-e5-large-instruct" not in source
    assert "harrier" not in source.lower()
    assert "SEA-LION-ModernBERT" not in source
    assert source.index("direct_student = distill(") < source.index("trainable.fit(")
    assert 'pca_dims=student_dimension if capacity_frontier else "auto"' in source
    assert "token_remove_pattern=token_remove_pattern" in source
    assert "direct_student.embedding.shape != (79_985, 1024)" in source
    assert 'candidate.get("vocabulary_size", 160_000)' in source


def test_worker_uses_same_parent_targets_and_full_width_int8_export():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()

    assert 'teacher_id != "aisingapore/SEA-LION-E5-Embedding-600M"' in source
    assert 'target_vectors.shape != (182_724, 1024)' in source
    assert "quantize_embeddings(best_weights, DType.Int8)" in source
    assert '"dimension": student_dimension' in source
    assert '"dtype": "int8"' in source
    # The gate is still applied, but asserting one exact source spelling broke
    # the moment the call was wrapped across lines.  Assert the call and its
    # argument separately instead.
    assert "validate_measured_artifact(" in source
    assert "hard_maximum_bytes=artifact_limit" in source


def test_worker_preserves_the_model_package_before_the_size_gate_can_raise():
    """The invariant whose violation destroyed a passing model.

    A size gate evaluated before the export was uploaded meant a gate failure
    was the reason the artifact was lost.  Order, not presence, is the thing
    worth testing: the gate may raise, so every byte it judges must already be
    in S3 when it runs.
    """
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()

    preserve_model = source.index('preserve.put(path, f"model/')
    run_gate = source.index("validate_measured_artifact(\n")
    assert preserve_model < run_gate, (
        "the model package must upload before the size gate runs; a gate that "
        "raises must never be the reason a model is lost"
    )


def test_worker_resolves_padding_from_the_static_tokenizer_api():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()

    assert "from model2vec.train.utils import get_probable_pad_token_id" in source
    assert "get_probable_pad_token_id(tokenlearn_student.tokenizer)" in source
    assert "tokenlearn_student.pad_token_id" not in source


def test_development_gate_precedes_heldout_scoring_and_model_upload():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()

    gate = source.index("development_gate = (")
    heldout = source.index('heldout = [row for row in annotations')
    upload = source.index('preserve.tree(export_dir, f"models/{candidate_id}")')
    assert gate < heldout < upload


def test_launcher_is_one_bounded_managed_job_without_endpoint():
    source = (ROOT / "scripts/start_multiwikiqa_direct_sealion.py").read_text()

    assert '"InstanceType": instance_type' in source
    assert 'os.environ.get("DIRECT_SEALION_INSTANCE_TYPE", "ml.g4dn.2xlarge")' in source
    assert '"MaxRuntimeInSeconds": maximum_runtime_seconds' in source
    assert 'os.environ.get("DIRECT_SEALION_MAX_RUNTIME_SECONDS", "10800")' in source
    assert '"no_endpoint": True' in source
    assert "create_endpoint" not in source
    # The single-job serialisation guard was replaced by an explicit allow-set
    # so that up to three concurrent jobs can run on distinct g4dn quotas.
    # What still must hold is that an *unexpected* active job refuses launch.
    assert "unexpected active SageMaker training job(s): " in source
    assert "refusing duplicate paid compute" in source
    assert "if not allowed or not active_names <= allowed:" in source
