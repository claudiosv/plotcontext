"""RC params for the ACM/ICML publication style."""

from typing import Any


def icml_rc_params(fig_x: float, fig_y: float) -> dict[str, Any]:
    """Build ACM/ICML rc params sized for a `fig_x` x `fig_y` inch figure."""
    return {
        "axes.labelsize": 9,
        "axes.titlepad": 0,
        "axes.titlesize": 9,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.hspace": 0.01,
        "figure.constrained_layout.use": True,  # Global toggle
        "figure.constrained_layout.w_pad": 0.02,
        "figure.dpi": 300,
        "figure.figsize": (fig_x, fig_y / fig_x),
        "font.family": "serif",
        "font.serif": ["Linux Libertine", "Libertine", "DejaVu Serif"],
        "font.size": 9,
        "legend.borderaxespad": 0,
        "legend.borderpad": 0,
        "legend.fontsize": 9,
        "lines.markersize": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": True,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
