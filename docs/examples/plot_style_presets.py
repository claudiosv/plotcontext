"""
Style presets
==============

``plot_context`` selects a seaborn context (``paper``, ``notebook``,
``talk``, ``poster``) or the bundled ``ieee`` style; ``PlotContext.slides()``-
equivalent rc params are available via ``PlotContext.slides_rc_params()`` for
a 16:9, presentation-scaled figure. Both scale fonts/lines/markers without
touching your data or plotting calls.
"""

# %%
import numpy as np
import pandas as pd

from plotcontext.polars_plot_context import PlotContext

rng = np.random.default_rng(2)
df = pd.DataFrame(
    {
        "epoch": np.arange(1, 21),
        "loss": 2.0 * np.exp(-0.15 * np.arange(1, 21)) + rng.normal(0, 0.02, 20),
    }
)

# %%
# A compact, print-oriented figure. As in the other examples, the
# ``__enter__``/``_draw``/``__exit__`` split (rather than a plain ``with``
# block) exists only so this docs gallery can capture each figure.
paper_ctx = PlotContext(
    title="Training loss (paper style)",
    x_label="Epoch",
    y_label="Loss",
    plot_context="paper",
)
paper_ctx.__enter__()
paper_ctx.ax.plot(df["epoch"], df["loss"])
paper_ctx._draw()

# %%
paper_ctx.__exit__(None, None, None)

# %%
# The same data, styled for a 16:9 presentation slide: larger fonts, thicker
# lines, higher DPI.
slides_ctx = PlotContext(
    title="Training loss (slides style)",
    x_label="Epoch",
    y_label="Loss",
    rc_params=PlotContext.slides_rc_params(),
)
slides_ctx.__enter__()
slides_ctx.ax.plot(df["epoch"], df["loss"])
slides_ctx._draw()

# %%
slides_ctx.__exit__(None, None, None)
