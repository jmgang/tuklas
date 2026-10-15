"""Read-only host checks and a repeat-call Cohere consistency probe on AWS."""
import json
from pathlib import Path
import shlex
import sys
import time
import boto3


def main(folder):
    folder = Path(folder)
    settings = json.loads((folder / "resources.json").read_text())
    marker = str(time.time_ns())
    session = boto3.Session(profile_name="admin-general", region_name=settings["region"])
    ssm = session.client("ssm")
    ids = [r["instance_id"] for r in settings["restored_volumes"]]
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": ids}])["InstanceInformationList"]
        if len(info) == 6 and all(i["PingStatus"] == "Online" for i in info):
            break
        time.sleep(5)
    else:
        raise RuntimeError("Not all restored hosts are SSM online")
    code = '''import json,pathlib,psutil,socket,subprocess
base=pathlib.Path('/opt/pycon-isolated')
assert (base/'bootstrap.json').exists() and (base/'ready.json').exists()
active=[p.pid for p in psutil.process_iter(['cmdline']) if any(a.endswith(('/run_isolated_aws.py','/isolated_model_http.py')) or ('pycon-' in a and a.endswith(('-worker.py','-controller.py'))) for a in (p.info['cmdline'] or []))]
assert not active,active
print(json.dumps({'host':socket.gethostname(),'ready':True,'active_workers':active,'disk':psutil.disk_usage(str(base))._asdict(),'units':subprocess.run(['systemctl','list-units','--all','--type=service','--no-legend','--plain'],capture_output=True,text=True).stdout}),flush=True)
'''
    command = ssm.send_command(InstanceIds=ids, DocumentName="AWS-RunShellScript", Parameters={
        "commands": [settings["python"] + " -u -c " + shlex.quote(code)], "executionTimeout": ["120"]})["Command"]["CommandId"]
    def collect(command, hosts, timeout=180):
        pending = set(hosts); records = {}; until = time.monotonic() + timeout
        while pending and time.monotonic() < until:
            for host in list(pending):
                try: result = ssm.get_command_invocation(CommandId=command, InstanceId=host)
                except ssm.exceptions.InvocationDoesNotExist: continue
                if result["Status"] not in ("Pending", "InProgress", "Delayed"):
                    records[host] = result; pending.remove(host)
            if pending: time.sleep(3)
        assert not pending, pending
        return records
    results = collect(command, ids)
    with (folder / ("restored-host-checks-" + marker + ".json")).open("x") as out:
        json.dump(results, out, indent=2, default=str)
    assert all(r["Status"] == "Success" for r in results.values()), results
    print("RESTORED_HOSTS_READY", ids, flush=True)
    host = settings["dedicated_models"]["cohere_v3"]["instance_id"]
    probe = '''import os,sys,json,numpy as np
from pathlib import Path
os.environ['PYCON_FRESH_ROOT']='/opt/pycon-isolated/input'
sys.path.insert(0,'/opt/pycon-isolated/code/scripts');sys.path.insert(0,'/opt/pycon-isolated/code/src')
import run_isolated_aws as w,run_fresh_aws_benchmark as shared
catalog=shared.read(Path('/opt/pycon-isolated/dedicated-benchmark/shared-catalog.json'))
corpus,_,_,_,_=shared.prepare()
assert catalog['corpus_texts_sha256']==shared.digest(corpus.texts)
values=np.load('/opt/pycon-isolated/dedicated-benchmark/cohere_v3/segment-vectors.npy',mmap_mode='r')
boundary=next((a+int(np.flatnonzero(np.linalg.norm(values[a:a+4096],axis=1)==0)[0]) for a in range(0,len(values),4096) if np.any(np.linalg.norm(values[a:a+4096],axis=1)==0)),len(values))
texts=[corpus.texts[o][a:b] for o,a,b in catalog['segments'][:96]]
_,client=w.session_clients();enc=w.CohereEncoder(client)
first,m1=enc.encode(texts,'corpus');second,m2=enc.encode(texts,'corpus')
def compare(a,b):
 cos=np.sum(a*b,axis=1);matches=np.argmax(a@b.T,axis=1)
 return {'minimum_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),'maximum_abs_difference':float(np.max(np.abs(a-b))),'strict_99999_passes':bool(np.all(cos>.99999)),'all_rows_match_own_index':bool(np.all(matches==np.arange(len(matches))))}
print(json.dumps({'identical_fresh_requests':compare(first,second),'fresh_vs_preserved':compare(first,values[:96]),'second_vs_preserved':compare(second,values[:96]),'completed_preserved_rows':boundary,'catalog_sha256':shared.file_digest(Path('/opt/pycon-isolated/dedicated-benchmark/shared-catalog.json')),'request_metadata':[m1,m2]}),flush=True)
'''
    command = ssm.send_command(InstanceIds=[host], DocumentName="AWS-RunShellScript", Parameters={
        "commands": ["PYTHONDONTWRITEBYTECODE=1 " + settings["python"] + " -u -c " + shlex.quote(probe)],
        "executionTimeout": ["300"]})["Command"]["CommandId"]
    with (folder / ("parity-probe-command-" + marker + ".json")).open("x") as out:
        json.dump({"command_id": command, "instance_id": host}, out)
    result = collect(command, [host], 330)[host]
    with (folder / ("parity-probe-result-" + marker + ".json")).open("x") as out:
        json.dump(result, out, indent=2, default=str)
    assert result["Status"] == "Success", result
    evidence = json.loads(result["StandardOutputContent"])
    with (folder / ("parity-evidence-" + marker + ".json")).open("x") as out:
        json.dump(evidence, out, indent=2)
    print("PARITY_EVIDENCE", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
