"""Run Task 2.4 against the frozen corpus vocabulary only."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.distillation import distill_custom


if __name__ == "__main__":
    print(distill_custom(ROOT, ROOT / "configs/models-v1.json", ROOT / "runs/custom-vocabulary-v1.json"))
