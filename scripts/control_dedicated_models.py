"""Exclusive model-host controller; preserve E5's unchanged active GPU build.

Static hosts are one-to-one with models. Cohere builds and measures on its own
client. All latency phases remain serial after a checksummed build barrier.
"""
from pathlib import Path
import hashlib
import json
import sys


def amended_source(source):
    begin = source.index('        # Both hosts are bootstrapped;')
    end = source.index('            gate = ', begin)
    block = '''        # Explicit host mapping is part of the recorded protocol.
        dedicated = settings["dedicated_models"]
        identifiers = [row["instance_id"] for row in dedicated.values()]
        assert len(identifiers)==len(set(identifiers))==5
        assert settings["client_instance"] not in identifiers
        original_prefix = settings["original_prefix"]
        def copy_file(prefix, relative):
            while True:
                try:
                    response = s3.get_object(Bucket=settings["bucket"], Key=prefix+"/benchmark/"+relative)
                    break
                except Exception as error:
                    code=getattr(error,"response",{}).get("Error",{}).get("Code")
                    if code not in ("404","NoSuchKey","NotFound"):raise
                    time.sleep(5)
            path = directory/relative; path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as out:
                while chunk:=response["Body"].read(1024*1024): out.write(chunk)
            assert shared.file_digest(path)==response["Metadata"]["sha256"],relative
            return path
        directory.mkdir(exist_ok=False)
        for filename in ("shared-catalog.json","shared-catalog.sha256.json","timing-queries.json"):
            path=copy_file(original_prefix,filename);publish(directory,path,settings)
        amendment={"dedicated_models":dedicated,"original_prefix":original_prefix,
            "frozen_encoder_code":settings["code_sha256"],
            "controller_wrapper_sha256":shared.file_digest(Path(__file__)),
            "teacher_build_retained_from_same_active_experiment":True,
            "old_static_builds_or_shared_cohere_outputs_reused":False,
            "dataset_changed":False,"frozen_models_changed":False,
            "latency_phases_serial":True,"shutdown_caps_extended":False,
            "static_host_type":"c7i.xlarge","static_host_vcpus":4}
        shared.save(directory/"dedicated-host-amendment.json",amendment)
        publish(directory,directory/"dedicated-host-amendment.json",settings)
        try:
            # Cohere is already independently building on its dedicated client;
            # E5 remains on the original client and dedicated GPU endpoint.
            for name in shared.NAMES:
                if name in ("teacher","cohere_v3"): continue
                row=dedicated[name]
                settings["server_instance"]=row["instance_id"]
                settings["server_private_ip"]=row["private_ip"]
                remote(["test -f /opt/pycon-isolated/ready.json"])
                start_server(name,"build")
                run("build",name)
                stop_server()
            shared.event(directory,stage="dedicated-static-builds-complete")
            for name,prefix in (("teacher",original_prefix),("cohere_v3",settings["prefix"])):
                while True:
                    try:
                        s3.head_object(Bucket=settings["bucket"],Key=prefix+"/benchmark/"+name+"/build.json")
                        break
                    except Exception as error:
                        code=getattr(error,"response",{}).get("Error",{}).get("Code")
                        if code not in ("404","NoSuchKey","NotFound"): raise
                    time.sleep(10)
                for filename in ("build.json","segment-vectors.npy","queries-vectors.npy","development-vectors.npy"):
                    path=copy_file(prefix,name+"/"+filename)
                    if name=="teacher":publish(directory,path,settings)
                record=shared.read(directory/name/"build.json")
                assert record["catalog_sha256"]==shared.file_digest(directory/"shared-catalog.json")
                for filename,checksum in record["files_sha256"].items():
                    assert shared.file_digest(directory/name/filename)==checksum
            # Teacher's original process must exit before timing starts; the
            # suspended old controller is never resumed.
            import psutil
            for process in psutil.process_iter(["cmdline"]):
                args=process.info["cmdline"] or []
                if any(x.endswith("/run_isolated_aws.py") for x in args) and "--phase" in args and args[args.index("--phase")+1]=="build":
                    process.wait(timeout=120)
            subprocess.run(["systemctl","stop","pycon-isolated-controller"],check=True)
            cohere_row=dedicated["cohere_v3"]
            settings["server_instance"]=cohere_row["instance_id"]
            settings["server_private_ip"]=cohere_row["private_ip"]
            remote(["systemctl show pycon-dedicated-cohere-build -p ActiveState -p ExecMainStatus",
                    "test $(systemctl show pycon-dedicated-cohere-build -p ExecMainStatus --value) = 0",
                    "test $(systemctl show pycon-dedicated-cohere-build -p ActiveState --value) != active"])
            shared.save(directory/"deployment-protocol.json",{**settings,"dedicated_host_amendment":amendment,
                "dataset_changed":False,"top_k":10,"old_outputs_reused":False,
                "teacher_fresh_vectors_retained_within_same_experiment":True})
            publish(directory,directory/"deployment-protocol.json",settings)
'''
    result=source[:begin]+block+source[end:]
    result=result.replace('"isolated-controller.lock"','"dedicated-controller.lock"')
    result=result.replace('''            label = phase + ("-"+name if name else "")
            argv =''','''            label = phase + ("-"+name if name else "")
            worker_settings_path=settings_path.parent/("dedicated-"+label+"-settings.json")
            shared.save(worker_settings_path,settings)
            argv =''')
    result=result.replace('"--settings", str(settings_path), "--phase", phase',
                          '"--settings", str(worker_settings_path), "--phase", phase')
    result=result.replace('settings_path.parent / (label+".log")',
                          'settings_path.parent / ("dedicated-"+label+".log")')
    marker='''            stop_server()
            current_unit = "pycon-isolated-"'''
    replacement='''            stop_server()
            row=settings["dedicated_models"][name]
            settings["server_instance"]=row["instance_id"]
            settings["server_private_ip"]=row["private_ip"]
            current_unit = "pycon-isolated-"'''
    assert marker in result
    result=result.replace(marker,replacement)
    result=result.replace('''                run("measure", name)
                stop_server()''','''                if name=="cohere_v3":
                    row=dedicated[name]
                    settings["server_instance"]=row["instance_id"]
                    settings["server_private_ip"]=row["private_ip"]
                    isolated_proof=shared.read(directory/name/"isolation-proof.json")
                    # Correct identity is preserved in a separate immutable record.
                    shared.save(directory/name/"dedicated-client-proof.json",{"client_instance":row["instance_id"],"inference":"Bedrock private endpoint"})
                    publish(directory,directory/name/"dedicated-client-proof.json",settings)
                    upload_gate = "import boto3,pathlib,hashlib; s=boto3.client('s3',region_name='ap-southeast-1'); r=s.get_object(Bucket="+repr(settings["bucket"])+",Key="+repr(settings["prefix"]+"/benchmark/all-builds-complete.json")+"); b=r['Body'].read(); assert hashlib.sha256(b).hexdigest()==r['Metadata']['sha256']; pathlib.Path('/opt/pycon-isolated/dedicated-benchmark/all-builds-complete.json').open('xb').write(b)"
                    commands=[settings["python"]+" -c "+shlex.quote(upload_gate),
                        "export PYCON_FRESH_ROOT="+shlex.quote(os.environ["PYCON_FRESH_ROOT"]),
                        "export PYCON_RESULT_BUCKET="+settings["bucket"],
                        "export PYCON_RESULT_PREFIX="+settings["prefix"]+"/benchmark",
                        "export PYTHONDONTWRITEBYTECODE=1",
                        " ".join(map(shlex.quote,[settings["python"],str(CODE/"scripts/run_isolated_aws.py"),"--directory",str(directory),"--settings","/opt/pycon-isolated/settings.json","--phase","measure","--model",name]))]
                    remote(commands,timeout=900)
                    copy_file(settings["prefix"],name+"/measurement.json")
                else:run("measure",name)
                stop_server()''')
    marker='''            download = f"""import boto3,pathlib,hashlib,json'''
    result=result.replace(marker,'''            row=dedicated["potion_retrieval"]
            settings["server_instance"]=row["instance_id"]
            settings["server_private_ip"]=row["private_ip"]
            download = f"""import boto3,pathlib,hashlib,json''')
    result=result.replace('report = {"protocol":','report = {"dedicated_hosts": dedicated, "protocol":')
    result=result.replace('"server_identity": identity, "client_instance": settings["client_instance"],',
        '"server_identity": identity, "client_instance": dedicated[name]["instance_id"] if name=="cohere_v3" else settings["client_instance"],')
    result=result.replace('CPU EC2 c7i.4xlarge','CPU EC2 c7i.xlarge')
    return result


if __name__=='__main__':
    settings=json.loads(Path(sys.argv[1]).read_text())
    path=Path(__file__).with_name('control_isolated_aws.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['code_sha256']['scripts/control_isolated_aws.py']
    exec(compile(amended_source(path.read_text()),str(Path(__file__)),'exec'),globals())
