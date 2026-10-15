"""Complete the explicitly resumed experiment without rebuilding good vectors.

Scoring workers run on dedicated AWS hosts. All bulk/scoring workers must exit
before serial, isolated latency measurement. Old files and outputs stay intact.
"""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main(settings_path):
    settings = json.loads(settings_path.read_text())
    code = Path("/opt/pycon-isolated/code")
    os.environ["PYCON_FRESH_ROOT"] = "/opt/pycon-isolated/input"
    os.environ["PYCON_RESULT_BUCKET"] = settings["bucket"]
    os.environ["PYCON_RESULT_PREFIX"] = settings["prefix"] + "/benchmark"
    sys.path.insert(0, str(code / "scripts")); sys.path.insert(0, str(code / "src"))
    import boto3
    import run_isolated_aws as worker
    import run_fresh_aws_benchmark as shared
    from pycon_model2vec_demo.grade3 import paired_with_teacher, TOP10_METRICS
    directory = Path(settings["output_directory"]); directory.mkdir(exist_ok=False)
    lock = (directory / "recovery-controller.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    s3 = boto3.client("s3", region_name=settings["region"])
    ssm = boto3.client("ssm", region_name=settings["region"])
    original_publish = worker.publish
    def publish_identical_or_new(output, path, config):
        key = config["prefix"] + "/benchmark/" + path.relative_to(output).as_posix()
        expected = shared.file_digest(path)
        try:
            existing = s3.head_object(Bucket=config["bucket"], Key=key)
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
            try:
                return original_publish(output, path, config)
            except Exception as conflict:
                if getattr(conflict, "response", {}).get("Error", {}).get("Code") != "PreconditionFailed":
                    raise
                existing = s3.head_object(Bucket=config["bucket"], Key=key)
        assert existing["Metadata"]["sha256"] == expected and existing["ContentLength"] == path.stat().st_size, key
    worker.publish = publish_identical_or_new
    dedicated = settings["dedicated_models"]
    assert len({r["instance_id"] for r in dedicated.values()}) == 5
    assert settings["client_instance"] not in {r["instance_id"] for r in dedicated.values()}
    for relative, expected in settings["code_sha256"].items():
        assert shared.file_digest(code / relative) == expected

    def remote(host, commands, timeout=180):
        reply = ssm.send_command(InstanceIds=[host], DocumentName="AWS-RunShellScript",
            Parameters={"commands": ["set -eu", *commands], "executionTimeout": [str(timeout)]},
            TimeoutSeconds=120, OutputS3BucketName=settings["bucket"], OutputS3KeyPrefix=settings["prefix"] + "/recovery/ssm-logs")
        command = reply["Command"]["CommandId"]; deadline = time.monotonic() + timeout + 120
        while time.monotonic() < deadline:
            try: record = ssm.get_command_invocation(CommandId=command, InstanceId=host)
            except ssm.exceptions.InvocationDoesNotExist: time.sleep(2); continue
            if record["Status"] == "Success": return record.get("StandardOutputContent", "")
            if record["Status"] not in ("Pending", "InProgress", "Delayed"):
                raise RuntimeError(json.dumps({"host": host, "command": command, "status": record["Status"],
                    "stdout": record.get("StandardOutputContent", "")[-2500:], "stderr": record.get("StandardErrorContent", "")[-2500:]}))
            time.sleep(3)
        raise TimeoutError(command)

    def exists(relative):
        try: s3.head_object(Bucket=settings["bucket"], Key=settings["prefix"] + "/benchmark/" + relative); return True
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"): return False
            raise

    def download(relative, source_prefix=None):
        path = directory / relative
        response = s3.get_object(Bucket=settings["bucket"], Key=(source_prefix or settings["prefix"]) + "/benchmark/" + relative)
        expected = response["Metadata"]["sha256"]
        if path.exists():
            assert shared.file_digest(path) == expected
            response["Body"].close()
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            for block in iter(lambda: response["Body"].read(1024 * 1024), b""): out.write(block)
        response["Body"].close(); assert shared.file_digest(path) == expected
        return path

    # A user-authorized continuation imports completed artifacts with original
    # checksums. It never recomputes successful quality results to replace them.
    previous = settings.get("previous_recovery_prefix")
    if previous:
        imported = {}
        for relative in ("shared-catalog.json", "shared-catalog.sha256.json", "timing-queries.json"):
            path = download(relative, previous); worker.publish(directory, path, settings)
            imported[relative] = shared.file_digest(path)
        for name in shared.LOCAL:
            for filename in ("quality.json", "build.json"):
                relative = name + "/" + filename
                path = download(relative, previous); worker.publish(directory, path, settings)
                imported[relative] = shared.file_digest(path)
        shared.save(directory / "continuation-provenance.json", {
            "source_prefix": previous, "files_sha256": imported,
            "dataset_changed": False, "model_weights_changed": False,
            "source_pause_snapshots": settings["source_pause_snapshots"]})
        worker.publish(directory, directory / "continuation-provenance.json", settings)
        shared.event(directory, stage="five-completed-quality-results-restored")

    def idle(host):
        check = "import psutil; active=[p.pid for p in psutil.process_iter(['cmdline']) if any(a.endswith(('/run_isolated_aws.py','/control_dedicated_models.py','/isolated_model_http.py')) or ('-worker.py' in a and a.startswith('/opt/pycon-isolated/')) for a in (p.info['cmdline'] or []))]; assert not active,active; print('workers idle')"
        return remote(host, [settings["python"] + " -c " + shlex.quote(check)])

    def wait_quality(name):
        last = 0
        while not exists(name + "/quality.json"):
            if time.monotonic() - last > 60:
                row = dedicated["potion_retrieval"] if name == "teacher" else dedicated[name]
                unit = settings["name"] + "-" + name
                output = remote(row["instance_id"], ["systemctl show " + unit + " -p ActiveState -p SubState -p ExecMainStatus",
                    "tail -5 /opt/pycon-isolated/" + unit + ".log"])
                if "ActiveState=failed" in output:
                    raise RuntimeError(name + " failed: " + output)
                shared.event(directory, stage="waiting-recovered-quality", model=name, status=output[-1200:])
                last = time.monotonic()
            time.sleep(5)
        return shared.read(download(name + "/quality.json"))

    results = {}
    for name in shared.LOCAL:
        if name == "teacher": continue
        results[name] = {"quality": wait_quality(name)}
        idle(dedicated[name]["instance_id"])
    # E5 vectors are already complete; score them on the same CPU search
    # hardware as the students, with Potion's worker finished and no server.
    potion = dedicated["potion_retrieval"]
    worker_script = "/opt/pycon-isolated/" + settings["name"] + "-worker.py"
    potion_settings = "/opt/pycon-isolated/" + settings["name"] + "-potion_retrieval-settings.json"
    command = [settings["python"], "-u", worker_script, "--settings", potion_settings, "--phase", "quality", "--model", "teacher"]
    if not exists("teacher/quality.json"):
        remote(potion["instance_id"], ["systemd-run --unit=" + settings["name"] + "-teacher --property=Restart=no " +
            "--property=StandardOutput=append:/opt/pycon-isolated/" + settings["name"] + "-teacher.log --property=StandardError=inherit " +
            "--setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OPENBLAS_NUM_THREADS=8 " + " ".join(map(shlex.quote, command))])
    results["teacher"] = {"quality": wait_quality("teacher")}
    idle(potion["instance_id"])
    for name in shared.LOCAL:
        download(name + "/build.json")
    shared.save(directory / "reports/five-model-quality.json", {"models": results, "scope": "Five completed models; Cohere and new latency pending",
        "query_count": 1127, "original_passage_count": 239704, "segment_count": 240765,
        "dataset_changed": False, "max_parent_cutoff": 10})
    worker.publish(directory, directory / "reports/five-model-quality.json", settings)
    shared.event(directory, stage="five-model-quality-complete", models=list(results))
    results["cohere_v3"] = {"quality": wait_quality("cohere_v3")}
    idle(dedicated["cohere_v3"]["instance_id"])
    download("cohere_v3/build.json")
    for relative in ("shared-catalog.json", "shared-catalog.sha256.json", "timing-queries.json"):
        download(relative)
    catalog_hash = shared.file_digest(directory / "shared-catalog.json")
    for name in shared.NAMES:
        quality = results[name]["quality"]
        assert quality["catalog_sha256"] == catalog_hash
        assert quality["summary"]["query_count"] == 1127
        assert len(quality["per_query"]) == 1127
        assert all(len(row["top10"]) == len({p["corpus_id"] for p in row["top10"]}) == 10 for row in quality["per_query"])
    gate = {"models": list(shared.NAMES), "builds": {name: shared.file_digest(directory / name / "build.json") for name in shared.NAMES},
        "bulk_and_quality_workers_idle_verified": True, "recovery_authorized": True}
    shared.save(directory / "all-builds-complete.json", gate); worker.publish(directory, directory / "all-builds-complete.json", settings)
    shared.event(directory, stage="recovery-all-builds-and-quality-complete")
    # Restore GPU serving only after the offline Cohere work completes, avoiding
    # idle GPU endpoint charges while waiting for the managed embedding API.
    if settings.get("restore_teacher_endpoint_after_builds"):
        sm = boto3.client("sagemaker", region_name=settings["region"])
        try:
            endpoint = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])
        except Exception as error:
            if "Could not find endpoint" not in str(error):
                raise
            sm.create_endpoint(EndpointName=settings["teacher_endpoint"],
                EndpointConfigName=settings["teacher_endpoint"],
                Tags=[{"Key": "Project", "Value": "pycon-davao-model2vec"},
                      {"Key": "RecoveryRun", "Value": settings["name"]}])
            shared.event(directory, stage="restored-gpu-endpoint-requested", endpoint=settings["teacher_endpoint"])
    active_unit = None
    active_host = None
    try:
        for name in shared.NAMES:
            for row in dedicated.values(): idle(row["instance_id"])
            row = dedicated.get(name)
            current = {**settings, "local_resource_id": settings["client_instance"]}
            if name not in ("teacher", "cohere_v3"):
                current.update(server_instance=row["instance_id"], server_private_ip=row["private_ip"])
                active_host = row["instance_id"]; active_unit = settings["name"] + "-serve-" + name
                command = [settings["python"], str(code / "scripts/isolated_model_http.py"), "--root", "/opt/pycon-isolated/input",
                    "--model", name, "--resource-id", row["instance_id"], "--bind", row["private_ip"]]
                remote(active_host, ["systemd-run --unit=" + active_unit + " --property=Restart=no " +
                    "--property=StandardOutput=append:/opt/pycon-isolated/" + active_unit + ".log --property=StandardError=inherit " +
                    " ".join(map(shlex.quote, command))])
                for attempt in range(60):
                    try: identity = worker.encoder(name, current).identity; break
                    except Exception:
                        if attempt == 59: raise
                        time.sleep(2)
            elif name == "teacher":
                sm = boto3.client("sagemaker", region_name=settings["region"])
                for attempt in range(180):
                    status = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])
                    if status["EndpointStatus"] == "InService": break
                    if status["EndpointStatus"] in ("Failed", "Deleting", "OutOfService"): raise RuntimeError(str(status))
                    time.sleep(10)
                else: raise TimeoutError("GPU endpoint did not become ready")
                identity = worker.encoder(name, current).identity
            else:
                current.update(server_instance=row["instance_id"], server_private_ip=row["private_ip"],
                    client_instance=row["instance_id"], local_resource_id=row["instance_id"])
                identity = worker.encoder(name, current, measured=True).identity
            proof = {"server_identity": identity, "client_instance": current["client_instance"],
                "server_instance": current["server_instance"], "all_bulk_and_quality_workers_idle": True,
                "model_hosts_distinct": True, "build_gate_sha256": shared.file_digest(directory / "all-builds-complete.json")}
            shared.save(directory / name / "isolation-proof.json", proof); worker.publish(directory, directory / name / "isolation-proof.json", settings)
            if name == "cohere_v3":
                command = [settings["python"], worker_script, "--settings", "/opt/pycon-isolated/" + settings["name"] + "-cohere_v3-settings.json", "--phase", "measure", "--model", name]
                remote(row["instance_id"], ["PYTHONDONTWRITEBYTECODE=1 " + " ".join(map(shlex.quote, command))], timeout=900)
                measurement = shared.read(download(name + "/measurement.json"))
            else:
                path = directory / (name + "-measurement-settings.json")
                shared.save(path, current)
                command = [settings["python"], str(code / "scripts/run_isolated_aws.py"), "--directory", str(directory),
                           "--settings", str(path), "--phase", "measure", "--model", name]
                with (directory / (name + "-measurement.log")).open("x") as out:
                    subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
                measurement = shared.read(directory / name / "measurement.json")
            results[name]["measurement"] = measurement
            if active_unit:
                remote(active_host, ["systemctl stop " + active_unit]); active_unit = None
            shared.event(directory, stage="recovered-model-measurement-complete", model=name)
    finally:
        if active_unit: remote(active_host, ["systemctl stop " + active_unit])

    report = {"protocol": settings, "models": results, "catalog_sha256": catalog_hash,
        "paired_with_teacher": {name: paired_with_teacher(results["teacher"]["quality"]["per_query"], results[name]["quality"]["per_query"],
             metrics=TOP10_METRICS) for name in shared.NAMES if name != "teacher"},
        "cohere_v4": {"status": "blocked", "reason": "Previously verified AWS Marketplace INVALID_PAYMENT_INSTRUMENT"},
        "limitations": "Post-hoc grade-3 cohort; incomplete pooled judgments; transductive students; static CPU versus GPU teacher deployment profiles; 160 single-request latency measurements; exact search, no ANN; restored verified vectors reused; Cohere provider RAM and model size unavailable; credits unverified."}
    shared.save(directory / "reports/combined-report.json", report)
    worker.publish(directory, directory / "reports/combined-report.json", settings)
    columns = ["Model", "Recall@1 %", "Recall@3 %", "Recall@5 %", "Recall@10 %", "MRR@10", "nDCG@10",
        "Client p50 ms", "Client p95 ms", "Client p99 ms", "Server encode p95 ms", "Search p95 ms", "Sampled server RAM MiB", "Model MiB", "Vectors MiB", "Successes", "Failures"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    def fmt(x, scale=1): return "unavailable" if x is None else f"{x * scale:.3f}"
    for name in shared.NAMES:
        q = results[name]["quality"]; m = results[name]["measurement"]; metrics = q["summary"]["overall"]
        rt = m["round_trip"] or {}; st = m["server_encode"] or {}
        cells = [name, *[fmt(metrics[k], 100) for k in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10")],
            fmt(metrics["mrr_at_10"]), fmt(metrics["ndcg_at_10"]), *[fmt(rt.get(k)) for k in ("p50_ms", "p95_ms", "p99_ms")],
            fmt(st.get("p95_ms")), fmt(q["search"]["p95_ms"]), fmt(m["server_max_rss_bytes"], 1 / 2**20),
            fmt(m["server"].get("model_bytes"), 1 / 2**20), fmt(q["fresh_build"]["vector_bytes"], 1 / 2**20),
            str(m["valid_successful_measurements"]), str(len(m["failures"]))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", report["limitations"], "", "Cohere v4 remains unavailable. Search latency is separate from encoding round-trip latency; percentiles are not added."])
    path = directory / "reports/combined-report.md"
    with path.open("x") as out: out.write("\n".join(lines) + "\n")
    worker.publish(directory, path, settings)
    shared.event(directory, stage="recovered-experiment-complete")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception as error:
        settings = json.loads(Path(sys.argv[1]).read_text())
        sys.path.insert(0, "/opt/pycon-isolated/code/scripts")
        import run_fresh_aws_benchmark as shared
        shared.event(Path(settings["output_directory"]), stage="recovery-controller-failed", error=repr(error))
        raise
