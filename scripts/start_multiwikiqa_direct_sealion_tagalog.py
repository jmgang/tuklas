#!/usr/bin/env python3
"""Launch the immutable Tagalog-only SEA-LION direct-child workflow."""

import os

os.environ["DIRECT_SEALION_RECIPE"] = "configs/multiwikiqa-direct-sealion-tagalog-v1.json"

from start_multiwikiqa_direct_sealion import main


if __name__ == "__main__":
    main()
