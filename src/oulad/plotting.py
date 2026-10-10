"""Shared matplotlib style and figure saving."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oulad.config import OUTCOME_COLORS


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
            "axes.axisbelow": True,
        }
    )


apply_style()


def save(fig: plt.Figure, path: Path | str) -> plt.Figure:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def outcome_color(label: str) -> str:
    return OUTCOME_COLORS.get(label, "#666666")
