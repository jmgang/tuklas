"""Pause automatic package maintenance on the six experiment hosts only."""
import json
from pathlib import Path
import shlex
import sys
import time
import boto3


REMOTE = r'''import json,pathlib,subprocess,psutil
units=['apt-daily.service','apt-daily-upgrade.service']
for unit in units:
    state=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip()
    assert state not in ('active','activating'), ('Wait for package transaction; never interrupt dpkg',unit,state)
assert not any(p.info['name'] in ('dpkg','unattended-upgrade','apt-get') for p in psutil.process_iter(['name']))
timers=['apt-daily.timer','apt-daily-upgrade.timer']
before={unit:subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip() for unit in units+timers}
subprocess.run(['systemctl','stop',*timers],check=True)
subprocess.run(['systemctl','mask','--runtime',*units,*timers],check=True)
# Additional protection against a manually invoked needrestart service restart.
path=pathlib.Path('/etc/needrestart/conf.d/99-pycon-benchmark-no-restart.conf')
body="$nrconf{override_rc}->{qr(^pycon-(?:repair|continue|isolated|dedicated).*)} = 0;\n"
if path.exists():
    assert path.read_text()==body
else:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as output:output.write(body)
subprocess.run(['perl','-c',str(path)],check=True)
after={unit:subprocess.run(['systemctl','is-enabled',unit],capture_output=True,text=True).stdout.strip() for unit in units+timers}
assert all(value=='masked' or value=='masked-runtime' for value in after.values()),after
assert not pathlib.Path('/run/systemd/shutdown/scheduled').exists()
print(json.dumps({'before':before,'after':after,'needrestart_exclusion':str(path),'shutdown_scheduled':False}))
'''


def main(folder):
    folder = Path(folder)
    settings = json.loads((folder / "resources.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    assert session.client("sts").get_caller_identity()["Account"] == settings["account"]
    ssm = session.client("ssm")
    ids = [r["instance_id"] for r in settings["restored_volumes"]]
    command = ssm.send_command(InstanceIds=ids, DocumentName="AWS-RunShellScript", Parameters={
        "commands": [settings["python"] + " -c " + shlex.quote(REMOTE)], "executionTimeout": ["120"]},
        OutputS3BucketName=settings["bucket"], OutputS3KeyPrefix=settings["prefix"] + "/recovery/maintenance")['Command']['CommandId']
    print("MAINTENANCE_COMMAND", command, flush=True)
    pending = set(ids); results = {}; until = time.monotonic()+180
    while pending and time.monotonic()<until:
        for host in list(pending):
            try:r=ssm.get_command_invocation(CommandId=command,InstanceId=host)
            except ssm.exceptions.InvocationDoesNotExist:continue
            if r['Status'] not in ('Pending','InProgress','Delayed'):
                results[host]=r;pending.remove(host)
        if pending:time.sleep(2)
    with (folder/('maintenance-results-'+command+'.json')).open('x') as output:
        json.dump(results,output,indent=2,default=str)
    assert not pending and all(r['Status']=='Success' for r in results.values()), results
    for host,r in results.items():print(host,r['StandardOutputContent'],flush=True)


if __name__=='__main__':
    main(sys.argv[1])
