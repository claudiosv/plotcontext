"""
Sketch mode
============

Passing ``sketch=True`` (or a config dict of ``scale``/``length``/
``randomness``) renders the figure in a hand-drawn, ``xkcd``-style, scoped
to just that plot -- global rcParams are restored on exit.
"""

# %%
import numpy as np
import pandas as pd

from plotcontext.polars_plot_context import PlotContext

rng = np.random.default_rng(1)
df = pd.DataFrame(
    {
        "day": np.arange(1, 15),
        "signups": np.cumsum(rng.integers(5, 20, size=14)),
    }
)

# %%
# ``__enter__``/``__exit__`` are called manually, and ``_draw()`` a cell
# early, purely so this docs gallery can capture the finished figure; in
# your own code just use ``with PlotContext(...) as ctx:``.
ctx = PlotContext(
    title="Daily signups",
    y_label="Cumulative signups",
    sketch={"scale": 2, "randomness": 3},
)
ctx.__enter__()
ctx.ax.plot(df["day"], df["signups"], marker="o")
ctx._draw()

# %%
ctx.__exit__(None, None, None)
