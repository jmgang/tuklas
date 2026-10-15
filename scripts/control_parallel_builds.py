"""Explicit scheduling amendment: retain active E5 build, start Cohere beside it.

No frozen model, input, encoder, or measurement code is changed. The original
controller is suspended before handoff; its child completes unchanged. Timings
remain serial and cannot start until both build processes have exited.
"""
from pathlib import Path
import hashlib
import json
import sys


def amended_source(source):
    start = source.index('        # Both hosts are bootstrapped;')
    end = source.index('            gate = ', start)
    replacement = '''        import psutil
        teacher_pid = int(sys.argv[2])
        original_pid = int(sys.argv[3])
        original = psutil.Process(original_pid)
        if original.status() != psutil.STATUS_STOPPED:
            raise RuntimeError("Original controller must be suspended")
        if (directory / "cohere_v3").exists():
            raise RuntimeError("Cohere already has outputs; refusing duplicate")
        for name in shared.NAMES:
            if name in ("teacher", "cohere_v3"): continue
            record = shared.read(directory / name / "build.json")
            assert record["catalog_sha256"] == shared.file_digest(directory / "shared-catalog.json")
            for filename, checksum in record["files_sha256"].items():
                assert shared.file_digest(directory / name / filename) == checksum
        teacher = psutil.Process(teacher_pid)
        args = teacher.cmdline()
        assert "--model" in args and args[args.index("--model")+1] == "teacher"
        assert "--phase" in args and args[args.index("--phase")+1] == "build"
        amendment = {"change": "Parallel E5 and Cohere corpus builds only",
            "original_controller_pid": original_pid, "teacher_worker_pid": teacher_pid,
            "original_controller_sha256": settings["code_sha256"]["scripts/control_isolated_aws.py"],
            "amended_wrapper_sha256": shared.file_digest(Path(__file__)),
            "dataset_changed": False, "models_changed": False,
            "measurement_parallel": False, "completed_fresh_builds_preserved": True,
            "shutdown_caps_unchanged": True}
        shared.save(directory / "parallel-build-amendment.json", amendment)
        publish(directory, directory / "parallel-build-amendment.json", settings)
        try:
            argv = [settings["python"], str(CODE / "scripts/run_isolated_aws.py"),
                "--directory", str(directory), "--settings", str(settings_path),
                "--phase", "build", "--model", "cohere_v3"]
            with (settings_path.parent / "parallel-build-cohere-v3.log").open("x") as out:
                cohere = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT)
                shared.event(directory, stage="parallel-builds-started", teacher_pid=teacher_pid,
                             cohere_pid=cohere.pid, measurement_workers_active=False)
                while teacher.is_running() and teacher.status() != psutil.STATUS_ZOMBIE:
                    if cohere.poll() not in (None, 0):
                        raise RuntimeError("Parallel Cohere build failed; preserved log")
                    time.sleep(5)
                # The original controller must not wake up and duplicate Cohere.
                subprocess.run(["systemctl", "stop", "pycon-isolated-controller"], check=True)
                if cohere.wait() != 0:
                    raise RuntimeError("Parallel Cohere build failed; preserved log")
            for name in ("teacher", "cohere_v3"):
                record = shared.read(directory / name / "build.json")
                assert record["catalog_sha256"] == shared.file_digest(directory / "shared-catalog.json")
                for filename, checksum in record["files_sha256"].items():
                    assert shared.file_digest(directory / name / filename) == checksum
                # Manifest must also have reached immutable S3 before timing.
                reply = s3.head_object(Bucket=settings["bucket"],
                    Key=settings["prefix"]+"/benchmark/"+name+"/build.json")
                assert reply["Metadata"]["sha256"] == shared.file_digest(directory / name / "build.json")
'''
    result = source[:start] + replacement + source[end:]
    result = result.replace('"isolated-controller.lock"', '"parallel-controller.lock"')
    result = result.replace('report = {"protocol":', 'report = {"scheduling_amendment": shared.read(directory / "parallel-build-amendment.json"), "protocol":')
    return result


if __name__ == "__main__":
    settings = json.loads(Path(sys.argv[1]).read_text())
    original = Path(__file__).with_name("control_isolated_aws.py")
    source = original.read_text()
    expected = settings["code_sha256"]["scripts/control_isolated_aws.py"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
    code = amended_source(source)
    compile(code, str(Path(__file__)), "exec")
    exec(compile(code, str(Path(__file__)), "exec"), globals())
