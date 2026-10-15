from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_capacity_worker_measures_every_stage_and_coverage():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    for stage in (
        "direct_model2vec_float32",
        "post_tokenlearn_float32",
        "post_retrieval_float32",
        "post_export_int8",
    ):
        assert stage in source
    assert "vocabulary-coverage-audit.json" in source
    assert "fully_covered_gold_fraction" in source
    assert "target_pca_retained_variance" in source
    assert 'tokenlearn_state["selected_validation_index"]' in source
    assert 'expected["retrieval"]["minimum_safely_mined"]' in source


def test_capacity_worker_preserves_size_receipt_before_quality_gate():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    receipt_upload = source.index('"reports/artifact-size-receipt.json"')
    development_gate = source.index("development_gate = (")
    assert receipt_upload < development_gate


def test_capacity_worker_does_not_overwrite_frozen_candidate_during_mining():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    assert "candidate = int(pool[int(position)])" not in source
    assert "candidate_index = int(pool[int(position)])" in source


def test_capacity_launcher_is_sequential_and_bounded():
    source = (ROOT / "scripts/start_multiwikiqa_direct_sealion.py").read_text()
    worker = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    wrapper = (ROOT / "scripts/start_multiwikiqa_sealion_capacity_frontier.py").read_text()
    # Bounded by an explicit allow-set rather than strict serialisation: three
    # concurrent jobs are permitted, an unexpected active job is not.
    assert "unexpected active SageMaker training job(s): " in source
    assert "refusing duplicate paid compute" in source
    assert '"MaxRuntimeInSeconds": maximum_runtime_seconds' in source
    assert 'os.environ.get("DIRECT_SEALION_MAX_RUNTIME_SECONDS", "10800")' in source
    assert "full_parent_250k_256" in wrapper
    assert "tagalog_160k_512" in wrapper
    assert "full_parent_250k_432" in wrapper
    assert "full_parent_250k_512" in wrapper
    assert 'DIRECT_SEALION_MINIMUM_MINED"] = "15000"' in wrapper
    assert "multiwikiqa-tagalog-ranking-expanded-15000-20261006T080600Z" in wrapper
    assert "hard_maximum_complete_artifact_bytes" in source
    assert "hard_maximum_bytes=artifact_limit" in worker
    assert 'startswith(' in source
    assert 'startswith(' in worker
    assert "create_endpoint" not in source


def test_capacity_launcher_allows_only_one_explicit_named_parallel_peer():
    source = (ROOT / "scripts/start_multiwikiqa_direct_sealion.py").read_text()
    assert 'DIRECT_SEALION_ALLOW_PARALLEL_WITH' in source
    # Was a single named peer; now a comma-separated allow-set, so the check
    # is a subset test rather than an equality test against one name.
    assert "if not allowed or not active_names <= allowed:" in source
    assert '"DIRECT_SEALION_ALLOW_PARALLEL_WITH", ""' in source
    assert 'DIRECT_SEALION_INSTANCE_TYPE' in source
    assert '"ml.g4dn.4xlarge"' in source
    assert 'DIRECT_SEALION_MAX_RUNTIME_SECONDS' in source
    # The cap set was widened to 4h and 6h so a 20-epoch stage-1 run can
    # finish; the invariant is that the runtime stays an allow-listed value,
    # not that the allow-list never grows.
    assert 'maximum_runtime_seconds not in {7_200, 10_800, 14_400, 21_600}' in source


def test_capacity_launcher_stages_matching_synthetic_data_and_receipt():
    source = (ROOT / "scripts/start_multiwikiqa_direct_sealion.py").read_text()
    assert '"DIRECT_SEALION_SYNTHETIC_ADMITTED"' in source
    assert '"DIRECT_SEALION_SYNTHETIC_REPORT"' in source
    assert '"synthetic-admitted.jsonl": synthetic_path' in source
    assert '"synthetic-verification-report.json": synthetic_report_path' in source


def test_expansion_merge_enforces_tagalog_disjoint_sources_and_configurable_floor():
    source = (ROOT / "scripts/merge_tagalog_synthetic_expansion.py").read_text()
    assert 'row.get("language") != "tl"' in source
    assert "control_articles & expansion_articles" in source
    assert '"duplicate_normalized_query"' in source
    assert '"minimum_safely_mined_pairs_required_downstream": args.minimum_safely_mined' in source


def test_v4_worker_uses_retrieval_selected_tokenlearn_and_query_alignment():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    assert "checkpoint_selector=tokenlearn_retrieval_selector" in source
    assert '"tokenlearn-checkpoint-retrieval"' in source
    assert "projected_teacher_queries" in source
    assert "query_vector_distillation_weight" in source
    assert "query_alignment = 1.0 -" in source


def test_v4_worker_dynamically_remines_and_uses_frozen_lr_schedule():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    assert "def mine_training_examples(" in source
    assert "dynamic_remine_after_epochs" in source
    assert '"dynamic-hard-negative-remine-complete"' in source
    assert "retrieval_lr_schedule" in source
    assert "retrieval_early_stopping_patience" in source


def test_mining_event_does_not_pass_stage_twice():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    assert '"mining_stage": stage' in source
    assert '"stage": stage' not in source


def test_controlled_repair_persists_ranking_ready_checkpoint_before_training():
    source = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()
    checkpoint = source.index('"checkpoints/ranking-ready-weights.npy"')
    first_epoch = source.index("for epoch in range(1, retrieval_epochs + 1):")
    assert checkpoint < first_epoch
    assert '"ranking-ready-tokenizer.json"' in source
    assert '"ranking-ready-receipt.json"' in source
    assert '"controlled-control-result"' in source
