"""Image-comparison tests that render real figures via pytest-mpl.

Run `pytest --mpl-generate-path=tests/baseline tests/test_visual.py` once to
(re)generate the reference images after an intentional visual change, then
verify with `pytest --mpl tests/test_visual.py`.
"""

import numpy as np
import pytest

from plotcontext.context_managers import plot_context
from plotcontext.plot_context_builder import Plot
from plotcontext.polars_plot_context import PlotContext
from plotcontext.single_plot_context import SinglePlotContext

# plotcontext.plot_context_builder.Plot mutates global rcParams via an
# unscoped sns.set_theme() call, so without pinning a style here these
# comparisons would depend on what other tests ran first in the session.
pytestmark = pytest.mark.mpl_image_compare(style="default")


def test_plot_context_manager_renders_sine_wave():
    with plot_context(figsize=(4, 3)) as (fig, ax):
        x = np.linspace(0, 2 * np.pi, 50)
        ax.plot(x, np.sin(x))
        ax.set_title("sine wave")
    return fig


def test_single_plot_context_renders_quadratic():
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with ctx as (_ctx, fig, ax):
        x = np.linspace(0, 10, 50)
        ax.plot(x, x**2)
        ctx.set_title("quadratic")
    return fig


def test_plot_context_renders_bar_chart():
    ctx = PlotContext(title="Bar chart", x_label="category", y_label="value")
    with ctx:
        ctx.ax.bar(["a", "b", "c"], [3, 7, 5])
    return ctx.figure


def test_plot_builder_renders_scatter():
    plot = Plot().title("Scatter").labels(x="x", y="y")
    with plot as p:
        p.ax.scatter([1, 2, 3], [3, 1, 2])
    return plot.figure
