"""Shared publication styling; imported only by reporting commands."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#273444",
            "text.color": "#273444",
            "axes.edgecolor": "#CFD7DD",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig, output, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
