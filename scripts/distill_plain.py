"""Run only the approved Task 2.3 plain-vocabulary distillation."""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pycon_model2vec_demo.distillation import distill_plain


if __name__ == "__main__":
    print(distill_plain(PROJECT_DIR, PROJECT_DIR / "configs" / "models-v1.json"))
