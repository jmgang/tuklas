"""Featurize unlabeled DisastIR passages, then train a fifth TokenLearn model."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.tokenlearn_training import featurize_corpus, train_custom_student


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("featurize", "train"))
    args = parser.parse_args()
    config = ROOT / "configs/tokenlearn-v1.json"
    if args.stage == "featurize":
        print(featurize_corpus(ROOT, config))
    else:
        print(train_custom_student(ROOT, config))
