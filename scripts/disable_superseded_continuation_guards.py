"""Disable only the two obsolete guards; retain the authorized cap unchanged."""
import json
from datetime import datetime, timezone
from pathlib import Path
import boto3


def main():
    folder = Path(__file__).resolve().parents[1] / "runs/pycon-continue-20261003-053452-4bc896"
    settings = json.loads((folder / "resources.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    assert session.client("sts").get_caller_identity()["Account"] == settings["account"]
    scheduler = session.client("scheduler")
    current = ["pycon-continue-20261003-053452-4bc896-cpu-stop",
               "pycon-continue-20261003-053452-4bc896-gpu-stop"]
    caps = {name: scheduler.get_schedule(Name=name) for name in current}
    for cap in caps.values():
        assert cap["State"] == "ENABLED"
        assert cap["ScheduleExpression"] == "at(2026-10-03T08:35:02)"
    ids = {row["instance_id"] for row in settings["restored_volumes"]}
    journal = []
    for name in ["pycon-resume-20261003-034045-7df262-ec2-stop",
                 "pycon-resume-20261003-034045-7df262-gpu-stop"]:
        before = scheduler.get_schedule(Name=name)
        assert before["ScheduleExpression"] == "at(2026-10-03T06:40:47)"
        payload = json.loads(before["Target"]["Input"])
        assert (set(payload.get("InstanceIds", [])) == ids or
                payload == {"EndpointName": settings["teacher_endpoint"]})
        parameters = {key: before[key] for key in (
            "Name", "GroupName", "ScheduleExpression", "ScheduleExpressionTimezone",
            "FlexibleTimeWindow", "Target", "Description", "KmsKeyArn", "StartDate",
            "EndDate", "ActionAfterCompletion") if key in before}
        parameters["State"] = "DISABLED"
        scheduler.update_schedule(**parameters)
        after = scheduler.get_schedule(Name=name)
        assert after["State"] == "DISABLED" and after["Target"] == before["Target"]
        journal.append({"name": name, "previous_state": before["State"],
                        "state": after["State"], "expression": after["ScheduleExpression"]})
    for name, before in caps.items():
        after = scheduler.get_schedule(Name=name)
        assert all(after[key] == before[key] for key in ("State", "Target", "ScheduleExpression"))
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "changes": journal,
              "current_cap_utc": "2026-10-03T08:35:02Z", "current_guards_unchanged": True,
              "instances_restarted": False}
    path = folder / ("obsolete-guards-disabled-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    with path.open("x") as output:
        json.dump(record, output, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
