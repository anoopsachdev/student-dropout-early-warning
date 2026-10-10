"""Run the full early-warning and fairness analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oulad.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
