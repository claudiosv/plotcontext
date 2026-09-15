"""
Fluent ``Plot`` builder
========================

``Plot`` is a zero-argument, chainable context manager: every option is set
through a fluent setter before the ``with`` block runs the actual seaborn/
matplotlib calls via ``p.sns`` / ``p.plt``.
"""

# %%
import numpy as np
import pandas as pd

from plotcontext.plot_context_builder import Plot

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "batch": np.tile(np.arange(1, 11), 2),
        "latency_ms": np.concatenate(
            [
                10 * np.arange(1, 11) * rng.uniform(0.9, 1.1, 10),
                4 * np.arange(1, 11) * rng.uniform(0.9, 1.1, 10),
            ]
        ),
        "model": ["baseline"] * 10 + ["optimized"] * 10,
    }
)

# %%
# ``__enter__``/``__exit__`` are called manually, and ``_draw()`` a cell
# early, purely so this docs gallery can capture the finished figure; in
# your own code, just use the chained builder inside a ``with`` block as
# shown in the module docstring above.
plot = (
    Plot()
    .title("Latency by batch size")
    .labels(x="Batch size", y="Latency (ms)")
    .log(x=True, y=True)
    .legend(outside=True)
)
plot.__enter__()
plot.sns.lineplot(data=df, x="batch", y="latency_ms", hue="model", marker="o")
plot._draw()
# Reserve room on the right for the outside legend (the automatic figure
# save this gallery performs doesn't know to grow the canvas for it).
plot.figure.subplots_adjust(right=0.78)

# %%
plot.__exit__(None, None, None)
