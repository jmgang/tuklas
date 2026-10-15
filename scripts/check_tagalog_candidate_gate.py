#!/usr/bin/env python3
"""Fast deterministic promotion gate for a Tagalog static student report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycon_model2vec_demo.tagalog_capacity_frontier import candidate_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = candidate_gate(json.loads(args.report.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
