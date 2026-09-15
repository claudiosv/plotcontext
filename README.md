# plotcontext

Scoped, chainable context managers for publication- and presentation-ready
[Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)
plots.

`plotcontext` wraps the usual `fig, ax = plt.subplots()` boilerplate in a
`with` block that applies consistent styling (fonts, palettes, spines,
legends, sketch mode, ...), proxies `plt`/`sns` calls so you don't have to
thread `ax=` through every call, and shows/closes/saves the figure for you on
exit.

## Installation

```bash
uv add plotcontext
```

or, with pip:

```bash
pip install plotcontext
```

## Quickstart

### Fluent builder (`Plot`)

```python
from plotcontext.plot_context_builder import Plot

with (
    Plot()
    .title("Latency")
    .labels(x="Batch", y="ms")
    .log(y=True)
    .legend(outside=True)
) as p:
    p.sns.lineplot(data=df, x="batch", y="ms", hue="model")
```

### Class-based context managers

```python
from plotcontext import PolarsPlotContext
from plotcontext.single_plot_context import SinglePlotContext

# Working with a polars.DataFrame
with PolarsPlotContext(title="Throughput", y_label="req/s") as ctx:
    ctx.sns.barplot(data=df, x="model", y="throughput")

# Backend-agnostic (pandas, numpy, ...)
with SinglePlotContext(style="whitegrid", plot_context="paper") as ctx:
    ctx.plt.plot(x, y)
```

Both context managers apply Seaborn styling on `__enter__`, inject the
managed `Axes` into every `ctx.plt.*` / `ctx.sns.*` call, and finalize
(labels, legend, grid, spines) and show/close the figure on `__exit__`.

### Standalone context managers

For quick, unstyled scoped plots:

```python
from plotcontext.context_managers import plot_context

with plot_context() as (fig, ax):
    ax.plot(x, y)
```

## Features

- **Presentation-ready styling** — `slides_rc_params()` tunes font sizes and
  figure size for 16:9 Google Slides decks; `plot_context="paper"` targets
  print/publication contexts.
- **Sketch mode** — flip on `plt.xkcd()`-style hand-drawn rendering with
  `sketch=True` or a `SketchConfig` (`scale`, `length`, `randomness`).
- **Style presets** — see
  [`plotcontext.styles`](https://github.com/claudiosv/plotcontext/tree/main/src/plotcontext/styles)
  for ICML, VS Code, and SciencePlots-compatible rc presets.
- **Boxplot annotation** — pass `annotate="x"` / `annotate="y"` to
  `ctx.sns.boxplot(...)` on a `PolarsPlotContext` to label grouped/hued
  boxplots with per-box counts, aligned or offset from the box center via
  `label_offset`.
- **Emoji tick labels** — render color emoji as x-axis tick images via
  `plotcontext.auto_emoji`.
- **Font tools** — `PlotContextFontTools` for consistent title/label/spine
  formatting outside of the full context-manager flow.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for all Python tooling.

```bash
uv sync              # install dependencies from the lockfile
uv run pytest        # run the test suite
uv run ruff check    # lint
```

## Documentation

Full API reference: https://claudiosv.github.io/plotcontext/

The published site is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).
The same `mkdocs.yml` also builds with [Zensical](https://zensical.org/), the newer static site
generator from the Material for MkDocs team, for side-by-side comparison:

```bash
uv run --group docs mkdocs serve                    # mkdocs-material, http://127.0.0.1:8000
uv run --group docs-zensical zensical serve          # Zensical, http://127.0.0.1:8000
uv run --group docs-zensical zensical build           # static build into site/
```

Zensical reads the existing `mkdocs.yml` directly, no separate config needed. By default it
renders with its own "modern" theme; add `variant: classic` under `theme` in `mkdocs.yml` to
match the Material for MkDocs look instead. Zensical is still young, so plugins beyond `search`,
`autorefs`, and `mkdocstrings` (Python handler) may not be supported yet — check
[its MkDocs compatibility docs](https://zensical.org/docs/compatibility/mkdocs/plugins/) before
relying on others.

## License

See
[LICENSES/SciencePlots.txt](https://github.com/claudiosv/plotcontext/blob/main/LICENSES/SciencePlots.txt)
for the license covering the bundled SciencePlots-derived style presets.
