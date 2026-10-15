"""Run the frozen replacement on its client EC2, controlling a different server.

Stages never overlap. Existing files mean stop, not resume or overwrite.
"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import pycon_model2vec_demo
import run_fresh_aws_benchmark as shared
from run_isolated_aws import encoder, publish
from pycon_model2vec_demo.grade3 import paired_with_teacher, TOP10_METRICS


def main(settings_path):
    import boto3
    settings = shared.read(settings_path)
    if settings["client_instance"] == settings["server_instance"]:
        raise ValueError("Different client/server instances required")
    directory = Path(settings["output_directory"])
    # This lock lives separately because catalog worker creates output directory.
    with (settings_path.parent / "isolated-controller.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ssm = boto3.client("ssm", region_name="ap-southeast-1")
        sm = boto3.client("sagemaker", region_name="ap-southeast-1")
        s3 = boto3.client("s3", region_name="ap-southeast-1")
        current_unit = None

        def remote(commands, timeout=1800):
            response = ssm.send_command(InstanceIds=[settings["server_instance"]],
                DocumentName="AWS-RunShellScript", Parameters={"commands": ["set -eu", *commands],
                    "executionTimeout": [str(timeout)]}, TimeoutSeconds=120,
                OutputS3BucketName=settings["bucket"], OutputS3KeyPrefix=settings["prefix"]+"/ssm-logs")
            command_id = response["Command"]["CommandId"]
            before = time.monotonic()
            while time.monotonic()-before < timeout+120:
                try:
                    record = ssm.get_command_invocation(CommandId=command_id,
                        InstanceId=settings["server_instance"])
                except ssm.exceptions.InvocationDoesNotExist:
                    time.sleep(2); continue
                if record["Status"] == "Success":
                    return record.get("StandardOutputContent", "")
                if record["Status"] not in ("Pending", "InProgress", "Delayed"):
                    raise RuntimeError(f"Server SSM {command_id}: {record['Status']} {record.get('StandardErrorContent','')[:2000]}")
                time.sleep(3)
            raise TimeoutError(f"Server command {command_id} did not finish")

        def assert_no_static_server():
            code = "import psutil; names=[p.pid for p in psutil.process_iter(['cmdline']) if any(x.endswith('/isolated_model_http.py') for x in (p.info['cmdline'] or []))]; assert not names, names; print('No static server active')"
            return remote([settings["python"] + " -c " + shlex.quote(code)])

        def stop_server():
            nonlocal current_unit
            if current_unit:
                remote(["systemctl stop " + shlex.quote(current_unit)])
                current_unit = None
            assert_no_static_server()

        def start_server(name, phase):
            nonlocal current_unit
            stop_server()
            current_unit = "pycon-isolated-" + name.replace("_", "-") + "-" + phase
            argv = [settings["python"], str(CODE / "scripts/isolated_model_http.py"),
                "--root", os.environ["PYCON_FRESH_ROOT"], "--model", name,
                "--resource-id", settings["server_instance"], "--bind", settings["server_private_ip"]]
            remote(["systemd-run --unit=" + current_unit + " --property=Restart=no " +
                    "--property=StandardOutput=append:/opt/pycon-isolated/" + current_unit + ".log " +
                    "--property=StandardError=inherit " + " ".join(map(shlex.quote, argv))])
            last = None
            for _ in range(60):
                try:
                    result = encoder(name, settings)
                    # Actual OS process count, not just declared identity.
                    check = "import psutil; p=[x for x in psutil.process_iter(['cmdline']) if any(a.endswith('/isolated_model_http.py') for a in (x.info['cmdline'] or []))]; assert len(p)==1, [x.pid for x in p]; print(p[0].pid)"
                    result.identity["ssm_verified_single_server_pid"] = remote([
                        settings["python"] + " -c " + shlex.quote(check)]).strip()
                    return result.identity
                except Exception as error:
                    last = error; time.sleep(2)
            raise RuntimeError(f"Static server not ready: {last!r}")

        def run(phase, name=None):
            label = phase + ("-"+name if name else "")
            argv = [settings["python"], str(CODE / "scripts/run_isolated_aws.py"),
                "--directory", str(directory), "--settings", str(settings_path), "--phase", phase]
            if name: argv += ["--model", name]
            with (settings_path.parent / (label+".log")).open("x") as out:
                result = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{label} failed; preserved log at {label}.log")

        # Both hosts are bootstrapped; no corpus building during model readiness.
        remote(["test -f /opt/pycon-isolated/ready.json"])
        before = time.monotonic()
        while time.monotonic()-before < 1800:
            status = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])
            if status["EndpointStatus"] == "InService": break
            if status["EndpointStatus"] in ("Failed", "Deleting", "OutOfService"):
                raise RuntimeError("GPU endpoint failed: " + str(status.get("FailureReason",status["EndpointStatus"])))
            time.sleep(10)
        else:
            raise TimeoutError("GPU endpoint did not become ready in 30 minutes")
        teacher_identity = encoder("teacher", settings).identity
        run("catalog")
        shared.save(directory / "deployment-protocol.json", {**settings, "teacher_identity": teacher_identity,
            "frozen_source_sha256": settings["code_sha256"], "dataset_changed": False,
            "old_outputs_reused": False, "top_k": 10})
        publish(directory, directory / "deployment-protocol.json", settings)
        try:
            for name in shared.NAMES:
                if name not in ("teacher", "cohere_v3"):
                    start_server(name, "build")
                run("build", name)
                if name not in ("teacher", "cohere_v3"):
                    stop_server()
            gate = {"models": list(shared.NAMES), "builds": {
                name: shared.file_digest(directory / name / "build.json") for name in shared.NAMES}}
            shared.save(directory / "all-builds-complete.json", gate)
            publish(directory, directory / "all-builds-complete.json", settings)
            shared.event(directory, stage="all-fresh-builds-complete")
            for name in shared.NAMES:
                if name not in ("teacher", "cohere_v3"):
                    identity = start_server(name, "measure")
                else:
                    stop_server(); identity = encoder(name, settings, measured=True).identity
                shared.save(directory / name / "isolation-proof.json", {
                    "active_stage": "measurement", "bulk_workers_active": False,
                    "build_gate_sha256": shared.file_digest(directory / "all-builds-complete.json"),
                    "server_identity": identity, "client_instance": settings["client_instance"],
                    "server_instance": settings["server_instance"], "stages_serial": True})
                publish(directory, directory / name / "isolation-proof.json", settings)
                run("measure", name)
                stop_server()
            # The serving EC2 now changes role to quality search; no serving or
            # latency measurement remains active on either machine.
            download = f"""import boto3,pathlib,hashlib,json
s=boto3.client('s3',region_name='ap-southeast-1'); root=pathlib.Path({str(directory)!r}); root.mkdir(exist_ok=False)
prefix={settings['prefix']!r}+'/benchmark/'
for page in s.get_paginator('list_objects_v2').paginate(Bucket={settings['bucket']!r},Prefix=prefix):
 for row in page.get('Contents',[]):
  relative=row['Key'][len(prefix):]
  if relative.startswith('events/'): continue
  path=root/relative; path.parent.mkdir(parents=True,exist_ok=True)
  response=s.get_object(Bucket={settings['bucket']!r},Key=row['Key']); h=hashlib.sha256()
  with path.open('xb') as out:
   while chunk:=response['Body'].read(1024*1024): h.update(chunk);out.write(chunk)
  assert h.hexdigest()==response['Metadata']['sha256'],relative
print('Fresh vectors and protocol downloaded and verified')
"""
            remote([settings["python"] + " -c " + shlex.quote(download)], timeout=900)
            server_settings = {**settings, "local_resource_id": settings["server_instance"]}
            create = "import pathlib,json; p=pathlib.Path('/opt/pycon-isolated/server-settings.json'); p.open('x').write(" + repr(json.dumps(server_settings)) + ")"
            remote([settings["python"] + " -c " + shlex.quote(create)])
            for name in shared.NAMES:
                commands = ["export PYCON_FRESH_ROOT=" + shlex.quote(os.environ["PYCON_FRESH_ROOT"]),
                    "export PYCON_RESULT_BUCKET=" + settings["bucket"],
                    "export PYCON_RESULT_PREFIX=" + settings["prefix"] + "/benchmark",
                    "export PYTHONDONTWRITEBYTECODE=1",
                    " ".join(map(shlex.quote,[settings["python"], str(CODE / "scripts/run_isolated_aws.py"),
                    "--directory", str(directory), "--settings", "/opt/pycon-isolated/server-settings.json",
                    "--phase", "quality", "--model", name]))]
                remote(commands, timeout=1800)
                response = s3.get_object(Bucket=settings["bucket"],
                    Key=settings["prefix"]+"/benchmark/"+name+"/quality.json")
                with (directory / name / "quality.json").open("xb") as out:
                    out.write(response["Body"].read())
            results = {name: {"quality": shared.read(directory / name / "quality.json"),
                "measurement": shared.read(directory / name / "measurement.json")} for name in shared.NAMES}
            report = {"protocol": shared.read(directory / "deployment-protocol.json"), "models": results,
                "paired_with_teacher": {name: paired_with_teacher(results["teacher"]["quality"]["per_query"],
                    results[name]["quality"]["per_query"], metrics=TOP10_METRICS) for name in shared.NAMES if name != "teacher"},
                "cohere_v4": {"status": "blocked", "reason": "Previously verified Marketplace INVALID_PAYMENT_INSTRUMENT"},
                "limitations": "Post-hoc grade-3 cohort; pooled incomplete relevance labels; transductive students; CPU-static vs GPU-teacher deployment profiles; 160 single-request warm latency samples; no ANN; credits unverified."}
            path = directory / "reports/combined-report.json"
            shared.save(path, report); publish(directory, path, settings)
            def fmt(value, scale=1):
                return "unavailable" if value is None else f"{value*scale:.2f}"
            headings = ["Model", "Serving", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
                "Client p50 ms", "Client p95 ms", "Client p99 ms", "Server encode p95 ms", "Search p95 ms",
                "Server sampled RAM MiB", "Model MiB", "Vectors MiB", "Measured successes", "Failed attempts"]
            lines = ["| " + " | ".join(headings) + " |", "|" + "---|"*len(headings)]
            for name, result in results.items():
                quality, op = result["quality"], result["measurement"]
                metrics=quality["summary"]["overall"]; rt=op["round_trip"] or {}; st=op["server_encode"] or {}
                cells=[name, "GPU SageMaker ml.g4dn.xlarge" if name=="teacher" else
                    "Managed Bedrock" if name=="cohere_v3" else "CPU EC2 c7i.4xlarge",
                    *(fmt(metrics[k],100)+"%" for k in ("recall_at_1","recall_at_3","recall_at_5","recall_at_10")),
                    fmt(metrics["mrr_at_10"]),fmt(metrics["ndcg_at_10"]),
                    *(fmt(rt.get(k)) for k in ("p50_ms","p95_ms","p99_ms")),fmt(st.get("p95_ms")),
                    fmt(quality["search"]["p95_ms"]),fmt(op["server_max_rss_bytes"],1/2**20),
                    fmt(op["server"].get("model_bytes"),1/2**20),fmt(quality["fresh_build"]["vector_bytes"],1/2**20),
                    str(op["valid_successful_measurements"]),str(len(op["failures"]))]
                lines.append("| " + " | ".join(cells) + " |")
            lines += ["", "Cohere v4 is unavailable: the previously verified Marketplace billing/access error remains unresolved.",
                "", report["limitations"], "", "Server encode excludes validation/transport; client latency includes the request round trip. Search is measured separately, not added to latency percentiles. RAM is the maximum sampled server RSS, not a guarantee of peak RAM. Cold loading and correctness smoke are recorded separately in server identity."]
            path=directory / "reports/combined-report.md"
            with path.open("x") as out: out.write("\n".join(lines)+"\n")
            publish(directory,path,settings)
            shared.event(directory, stage="isolated-benchmark-complete")
        finally:
            stop_server()


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception as error:
        directory = Path(shared.read(Path(sys.argv[1]))["output_directory"])
        if directory.exists(): shared.event(directory, stage="isolated-controller-failed", error=repr(error))
        print(repr(error), flush=True)
        raise
