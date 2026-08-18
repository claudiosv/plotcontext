"""RC params tuned for the standard 16:9 presentation slide layout."""

from typing import Any

SLIDES_RC: dict[str, Any] = {
    # 16:9 aspect ratio matching standard slide dimensions
    "figure.figsize": (10.0, 5.625),
    "figure.dpi": 300,
    # Base font scaling
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    # Tick scaling
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    # Legend scaling
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    # Line and marker visibility
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "axes.linewidth": 1.5,
    # Export settings
    "savefig.bbox": "tight",
    "savefig.transparent": True,
}
