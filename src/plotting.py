"""Shared matplotlib style: recessive grid/axes, ink-coloured text, validated palette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import COLORS, FIG_DIR  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
    "savefig.facecolor": COLORS["surface"], "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["ink2"], "xtick.color": COLORS["ink2"], "ytick.color": COLORS["ink2"],
    "text.color": COLORS["ink"], "axes.titlecolor": COLORS["ink"], "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10,
    "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name: str, folder=None):
    folder = folder or FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)
