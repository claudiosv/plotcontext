"""Standalone context managers for scoped matplotlib/seaborn figure handling."""

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns


@contextmanager
def plot_context(*args, **kwargs) -> Iterator[tuple[plt.Figure, plt.Axes]]:
    """Yields a (fig, ax) tuple, then shows and closes the figure."""
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.show()
        plt.close(fig)


@contextmanager
def auto_show() -> Iterator[None]:
    """Wraps plot execution, calling show() and closing all figures on exit."""
    try:
        yield
    finally:
        plt.show()
        plt.close("all")


@contextmanager
def facet_context(
    g: sns.FacetGrid,
) -> Iterator[tuple[sns.FacetGrid, plt.Figure, plt.Axes | list[plt.Axes]]]:
    """Manages a Seaborn figure-level plot, yielding the grid, figure, and axes."""
    try:
        # g.ax works for single-plot grids; g.axes is needed for multi-plot grids
        axes = g.ax if hasattr(g, "ax") else g.axes
        yield g, g.fig, axes
    finally:
        plt.show()
        plt.close(g.fig)
