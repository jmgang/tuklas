#!/usr/bin/env python3
"""Launch one frozen Tagalog SEA-LION capacity-frontier candidate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "candidate",
        choices=(
            "full_parent_250k_256",
            "tagalog_160k_512",
            "full_parent_250k_432",
            "full_parent_250k_512",
        ),
    )
    args = parser.parse_args()
    recipe_by_candidate = {
        "full_parent_250k_256": "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v1.json",
        "tagalog_160k_512": "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v1.json",
        "full_parent_250k_432": "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v2.json",
        "full_parent_250k_512": "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v5-512.json",
    }
    os.environ["DIRECT_SEALION_RECIPE"] = recipe_by_candidate[args.candidate]
    os.environ["DIRECT_SEALION_CANDIDATE"] = args.candidate
    if args.candidate == "full_parent_250k_512":
        expanded = ROOT / "runs/multiwikiqa-tagalog-ranking-expanded-15000-20261006T080600Z"
        os.environ["DIRECT_SEALION_SYNTHETIC_ADMITTED"] = str(expanded / "admitted.jsonl")
        os.environ["DIRECT_SEALION_SYNTHETIC_REPORT"] = str(expanded / "verification-report.json")
        os.environ["DIRECT_SEALION_MINIMUM_MINED"] = "15000"
    from start_multiwikiqa_direct_sealion import main as launch

    launch()


if __name__ == "__main__":
    main()
