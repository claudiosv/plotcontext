"""
Boxplot count annotations
==========================

``PolarsPlotContext`` can annotate a grouped/hued boxplot with the number of
observations in each box, either centered on the box or offset to the side
via ``label_offset``.
"""

# %%
import polars as pl

from plotcontext import PolarsPlotContext

df = pl.DataFrame(
    {
        "group": ["a"] * 6 + ["b"] * 6,
        "sub": (["x", "x", "x", "y", "y", "y"]) * 2,
        "value": [1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7],
    }
)

# %%
# The context manager is normally used as ``with PolarsPlotContext(...) as
# ctx:``. Here ``__enter__``/``__exit__`` are called manually, and finalizing
# (title/labels/legend) is triggered a cell early, purely so this docs
# gallery can capture the finished figure between build steps.
ctx = PolarsPlotContext(title="Count-annotated boxplot", y_label="value")
ctx.__enter__()
ctx.sns.boxplot(
    data=df,
    x="group",
    y="value",
    hue="sub",
    annotate="x",
    label_offset=0.2,
)
ctx._draw()

# %%
ctx.__exit__(None, None, None)
