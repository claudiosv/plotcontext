"""Fluent, scoped plotting context for seaborn/matplotlib.

Zero-kwarg constructor; all configuration via chainable setters.

Examples
--------
>>> with (
...     Plot()
...     .title("Latency")
...     .labels(x="Batch", y="ms")
...     .log(y=True)
...     .legend(outside=True) as p
... ):
...     p.sns.lineplot(data=df, x="batch", y="ms", hue="model")
"""

from __future__ import annotations

import datetime
import inspect
import io
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SeabornStyle = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]
SeabornContext = Literal["paper", "notebook", "talk", "poster", "ieee"]
GridAxis = Literal["both", "x", "y"]

SLIDES_RC: dict[str, Any] = {
    # 16:9 aspect ratio matching standard slide dimensions
    "figure.figsize": (10.0, 5.625),
    "figure.dpi": 300,
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "axes.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
}

SKETCH_FONTS_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["xkcd Script", "Comic Sans MS", "Arial"],
    "font.serif": ["xkcd Script", "Comic Sans MS", "Arial"],
}


class SketchConfig(TypedDict, total=False):
    """Keyword arguments forwarded to ``plt.xkcd``.

    Attributes
    ----------
    scale : float
        Amplitude of the wiggle perpendicular to the line.
    length : float
        Wiggle wavelength along the line.
    randomness : float
        Scale factor of the wiggle randomness.
    """

    scale: float
    length: float
    randomness: float


class ModuleProxy:
    """Proxy a module so callables are routed through the owning ``Plot``.

    Parameters
    ----------
    module : Any
        Module to proxy (e.g. ``seaborn`` or ``matplotlib.pyplot``).
    context : Plot
        Owning plot context used for kwarg injection.
    """

    def __init__(self, module: Any, context: Plot) -> None:
        self._module = module
        self._context = context

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._context._inject_and_call(attr, *args, **kwargs)

            return wrapper
        return attr


class Plot(AbstractContextManager["Plot"]):  # noqa: PLR0904
    """Fluent context manager for a seaborn-styled matplotlib figure.

    All configuration happens through chainable setters; the constructor
    takes no arguments. Style layers (seaborn theme, rc params, palette,
    sketch mode) are pushed onto an ``ExitStack`` on ``__enter__`` and
    fully reverted on ``__exit__``, so nothing leaks into global state.

    Examples
    --------
    >>> plot = (
    ...     Plot()
    ...     .slides()
    ...     .title("Throughput")
    ...     .labels(x="Threads", y="Ops/s")
    ...     .grid("y")
    ...     .legend(top=True)
    ... )
    >>> with plot as p:
    ...     p.sns.barplot(data=df, x="threads", y="ops", hue="impl")
    """

    _NO_AX_FUNCS: frozenset[str] = frozenset({
        "relplot",
        "catplot",
        "move_legend",
        "FacetGrid",
    })

    def __init__(self) -> None:
        # Decoration
        self._title: str | dict[str, Any] | None = None
        self._xlabel: str | None = None
        self._ylabel: str | None = None
        self._despine: bool = False
        self._grid_axis: GridAxis | None = None
        self._legend_kwargs: dict[str, Any] | None = None
        self._legend_outside: bool = False
        self._legend_top: bool = False
        self._log_x: bool = False
        self._log_y: bool = False
        self._formatter_x: Any | None = None
        self._formatter_y: Any | None = None
        self._tight: bool = False

        # Styling
        self._context: SeabornContext = "notebook"
        self._style: SeabornStyle | dict[str, Any] = "whitegrid"
        self._extra_styles: list[str] = []
        self._palette_name: str = "colorblind"
        self._color_map_spec: dict[Any, int] | None = None
        self._color_map: dict[Any, tuple[float, ...]] | None = None
        self._font_scale: float = 1.0
        self._rc: dict[str, Any] = {}
        self._sketch: SketchConfig | None = None

        # Figure geometry
        self._fig_x: float | None = None
        self._fig_y: float | None = None
        self._fig_scale: float = 1.0
        self._dpi: int | None = None
        self._create_fig: bool = True

        # State
        self.figure: Figure | None = None
        self.ax: Axes | None = None
        self.exited: bool = False
        self._drawn: bool = False
        self._debug: bool = False
        self._stack = ExitStack()

    # ------------------------------------------------------------------ #
    # Fluent setters                                                     #
    # ------------------------------------------------------------------ #

    def title(self, text: str, **kwargs: Any) -> Self:
        """Set the axes title.

        Parameters
        ----------
        text : str
            Title text.
        **kwargs
            Extra keyword arguments forwarded to ``Axes.set_title``.

        Returns
        -------
        Self
        """
        self._title = {"label": text, **kwargs} if kwargs else text
        return self

    def labels(self, x: str | None = None, y: str | None = None) -> Self:
        """Set axis labels.

        Parameters
        ----------
        x, y : str, optional
            Labels for the x and y axes.

        Returns
        -------
        Self
        """
        if x is not None:
            self._xlabel = x
        if y is not None:
            self._ylabel = y
        return self

    def size(
        self,
        x: float | None = None,
        y: float | None = None,
        scale: float = 1.0,
    ) -> Self:
        """Set figure size in inches, with an optional uniform scale.

        Parameters
        ----------
        x, y : float, optional
            Width and height. Defaults fall back to rc ``figure.figsize``.
        scale : float
            Multiplier applied to both dimensions.

        Returns
        -------
        Self
        """
        self._fig_x = x
        self._fig_y = y
        self._fig_scale = scale
        return self

    def dpi(self, dpi: int) -> Self:
        """Set figure DPI."""
        self._dpi = dpi
        return self

    def context(self, context: SeabornContext) -> Self:
        """Set the seaborn plotting context (or ``"ieee"`` for SciencePlots)."""
        self._context = context
        return self

    def style(self, style: SeabornStyle | dict[str, Any]) -> Self:
        """Set the seaborn axes style."""
        self._style = style
        return self

    def extra_styles(self, *names: str) -> Self:
        """Layer additional matplotlib style sheets on top."""
        self._extra_styles.extend(names)
        return self

    def palette(
        self, name: str = "colorblind", mapping: dict[Any, int] | None = None
    ) -> Self:
        """Set the color palette and an optional category-to-index map.

        Parameters
        ----------
        name : str
            Seaborn palette name.
        mapping : dict, optional
            Maps category values to palette indices; injected as
            ``palette=`` into plotting calls for stable colors across plots.

        Returns
        -------
        Self
        """
        self._palette_name = name
        self._color_map_spec = mapping
        return self

    def font_scale(self, scale: float) -> Self:
        """Scale all seaborn-managed font sizes."""
        self._font_scale = scale
        return self

    def rc(self, params: dict[str, Any]) -> Self:
        """Merge matplotlib rc params, scoped to the context."""
        self._rc.update(params)
        return self

    def slides(self) -> Self:
        """Apply the 16:9 Google-Slides preset (large fonts, 300 dpi)."""
        return self.rc(SLIDES_RC)

    def sketch(self, **config: Any) -> Self:
        """Enable xkcd sketch mode.

        Parameters
        ----------
        **config
            Optional ``scale``, ``length``, ``randomness`` forwarded to
            ``plt.xkcd``.

        Returns
        -------
        Self
        """
        self._sketch = config  # type: ignore[assignment]
        return self

    def legend(
        self, *, outside: bool = False, top: bool = False, **kwargs: Any
    ) -> Self:
        """Enable and position the legend.

        Parameters
        ----------
        outside : bool
            Anchor the legend outside the axes (right side).
        top : bool
            Anchor the legend above the axes, centered.
        **kwargs
            Forwarded to ``Axes.legend``.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If both `outside` and `top` are set.
        """
        if outside and top:
            msg = "Legend cannot be both outside and top"
            raise ValueError(msg)
        self._legend_kwargs = kwargs
        self._legend_outside = outside
        self._legend_top = top
        return self

    def grid(self, axis: GridAxis = "both") -> Self:
        """Enable a dashed grid on the given axis."""
        self._grid_axis = axis
        return self

    def despine(self) -> Self:
        """Remove top/right/left spines via ``sns.despine(left=True)``."""
        self._despine = True
        return self

    def log(
        self,
        *,
        x: bool = False,
        y: bool = False,
        fmt_x: Any | None = None,
        fmt_y: Any | None = None,
    ) -> Self:
        """Use logarithmic scale with plain (non-scientific) tick labels.

        Parameters
        ----------
        x, y : bool
            Which axes to set to log scale.
        fmt_x, fmt_y : matplotlib formatter, optional
            Custom major formatters; default is ``ScalarFormatter``.

        Returns
        -------
        Self
        """
        self._log_x = self._log_x or x
        self._log_y = self._log_y or y
        self._formatter_x = fmt_x or self._formatter_x
        self._formatter_y = fmt_y or self._formatter_y
        return self

    def tight(self) -> Self:
        """Apply ``plt.tight_layout`` on exit."""
        self._tight = True
        return self

    def debug(self) -> Self:
        """Log injection decisions and call results."""
        self._debug = True
        return self

    def no_figure(self) -> Self:
        """Skip creating a figure/axes on enter (e.g. for figure-level plots)."""
        self._create_fig = False
        return self

    # ------------------------------------------------------------------ #
    # Proxies                                                            #
    # ------------------------------------------------------------------ #

    @property
    def sns(self) -> Any:
        """Seaborn proxy with automatic ``ax``/``palette`` injection."""
        return ModuleProxy(sns, self)

    @property
    def plt(self) -> Any:
        """Pyplot proxy with automatic ``ax``/``palette`` injection."""
        return ModuleProxy(plt, self)

    # ------------------------------------------------------------------ #
    # Context management                                                 #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> Self:
        plt.ion()
        self._setup_styles()

        palette = sns.color_palette(self._palette_name)
        if self._color_map_spec:
            self._color_map = {k: palette[v] for k, v in self._color_map_spec.items()}

        if self._create_fig:
            rc_x, rc_y = self._rc.get("figure.figsize") or rcParams.get(
                "figure.figsize", (10.0, 5.625)
            )
            fig_x = (self._fig_x or rc_x) * self._fig_scale
            fig_y = (self._fig_y or rc_y) * self._fig_scale
            self.figure, self.ax = plt.subplots(figsize=(fig_x, fig_y))
        return self

    def _setup_styles(self) -> None:
        """Push all style layers onto the exit stack for clean teardown."""
        if self._context == "ieee":
            self._stack.enter_context(plt.style.context(["science", "ieee"]))
            return

        sns.set_theme(
            context=self._context, style=self._style, palette=self._palette_name
        )

        contexts: list[AbstractContextManager[Any]] = [
            sns.plotting_context(context=self._context, font_scale=self._font_scale),
            sns.axes_style(self._style),
            sns.color_palette(self._palette_name),
            plt.rc_context(
                {**self._rc, **SKETCH_FONTS_RC}
                if self._sketch is not None
                else self._rc
            ),
        ]

        if self._extra_styles:
            contexts.append(plt.style.context(self._extra_styles))
        if self._sketch is not None:
            contexts.append(plt.xkcd(**self._sketch))

        for ctx in contexts:
            self._stack.enter_context(ctx)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                if self.ax is None:
                    warnings.warn("No ax found on exit", stacklevel=2)
                else:
                    self._draw()
                plt.show()
        finally:
            self._close()
            self._stack.close()
            self.exited = True

    # ------------------------------------------------------------------ #
    # Injection machinery                                                #
    # ------------------------------------------------------------------ #

    def _accepts_kwarg(self, func: Callable[..., Any], name: str) -> bool:
        func_name = getattr(func, "__name__", repr(func))
        if name == "ax" and func_name in self._NO_AX_FUNCS:
            return False

        try:
            sig = inspect.signature(func)
        except ValueError:
            return False

        accepts = name in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if self._debug:
            print(f"{func_name} accepts {name!r}: {accepts}")
        return accepts

    def _inject_and_call(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if self._accepts_kwarg(func, "ax") and kwargs.get("ax") is None:
            if self.ax is None:
                self.ax = plt.gca()
            kwargs["ax"] = self.ax

        if (
            self._color_map
            and self._accepts_kwarg(func, "palette")
            and kwargs.get("palette") is None
        ):
            kwargs["palette"] = self._color_map

        result = func(*args, **kwargs)

        if self._debug:
            shown = {k: v for k, v in kwargs.items() if k != "data"}
            print(f"Called {func.__name__} kwargs={shown} -> {type(result).__name__}")

        if isinstance(result, Axes):
            self.ax = result
        elif isinstance(getattr(result, "axes", None), Axes):
            self.ax = result.axes

        return result

    # ------------------------------------------------------------------ #
    # Finalization                                                       #
    # ------------------------------------------------------------------ #

    def _draw(self) -> None:
        """Apply deferred decoration (labels, title, scales, legend, grid)."""
        if self.ax is None:
            return

        if self._xlabel:
            self.ax.set_xlabel(self._xlabel)
        if self._ylabel:
            self.ax.set_ylabel(self._ylabel)

        if self._title is not None:
            if isinstance(self._title, dict):
                self.ax.set_title(**self._title)
            else:
                self.ax.set_title(self._title)

        if self._despine:
            sns.despine(left=True)

        if self._log_x:
            self.ax.set(xscale="log")
            self.ax.xaxis.set_major_formatter(self._formatter_x or ScalarFormatter())
        if self._log_y:
            self.ax.set(yscale="log")
            self.ax.yaxis.set_major_formatter(self._formatter_y or ScalarFormatter())

        if self._legend_kwargs is not None:
            legend = self.ax.legend(**self._legend_kwargs)
            if self._legend_top:
                legend.set_bbox_to_anchor((0.5, 1.15))
                legend.set_loc(9)  # upper center
            elif self._legend_outside:
                legend.set_bbox_to_anchor((1.05, 1.0))
                legend.borderaxespad = 0.0
            if self._sketch is not None:
                for text in legend.get_texts():
                    text.set_fontfamily("xkcd Script")

        if self._grid_axis is not None:
            self.ax.grid(
                visible=True, linestyle="dashed", alpha=0.6, axis=self._grid_axis
            )

        if self._tight:
            plt.tight_layout()

        self._drawn = True

    def _close(self) -> None:
        if mpl.get_backend().startswith("module://ipympl"):
            return
        if self.figure is not None:
            plt.close(self.figure)

    # ------------------------------------------------------------------ #
    # Export                                                             #
    # ------------------------------------------------------------------ #

    def _require_figure(self) -> Figure:
        if self.figure is None:
            msg = (
                "No figure to export. Ensure you're within (or have used) the context."
            )
            raise RuntimeError(msg)
        if not self._drawn:
            self._draw()
        return self.figure

    def save(self, filename: Path | str, **save_kwargs: Any) -> None:
        """Save the figure (PDF by default).

        Parameters
        ----------
        filename : Path or str
            Output path.
        **save_kwargs
            Forwarded to ``Figure.savefig``; merged over the defaults.
        """
        fig = self._require_figure()
        kwargs: dict[str, Any] = {"format": "pdf", **save_kwargs}
        fig.savefig(Path(filename), **kwargs)
        print(f"Saved figure to {filename} with kwargs: {kwargs}")

    def save_fig(self, output_path: Path | str, name: str) -> None:
        """Save timestamped PDF and transparent PNG copies.

        Parameters
        ----------
        output_path : Path or str
            Directory to save into (created if missing).
        name : str
            Base filename; ``/`` characters are replaced with ``_``.

        Raises
        ------
        ValueError
            If `output_path` exists and is not a directory.
        """
        output_path = Path(output_path)
        if output_path.exists() and not output_path.is_dir():
            msg = f"Output path {output_path} exists and is not a directory"
            raise ValueError(msg)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.exited:
            warnings.warn(
                "save_fig invoked before exit, figure might be incomplete", stacklevel=2
            )

        name = name.strip().replace("/", "_")
        stamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d-%H-%M-%S")
        fig = self.figure or plt.gcf()

        fig.savefig(output_path / f"{name}_{stamp}.pdf")
        fig.savefig(output_path / f"{name}_{stamp}.png", transparent=True)

    def clip(self, **save_kwargs: Any) -> None:
        """Copy the figure to the macOS clipboard as a PNG.

        Parameters
        ----------
        **save_kwargs
            Forwarded to ``savefig``; merged over transparent 300-dpi defaults.
        """
        import clipin  # noqa: PLC0415 - optional macOS-only dependency

        fig = self._require_figure()
        kwargs: dict[str, Any] = {
            "format": "png",
            "dpi": 300,
            "bbox_inches": "tight",
            "transparent": True,
            **save_kwargs,
        }
        with io.BytesIO() as buffer:
            fig.savefig(buffer, **kwargs)
            clipin.copy(buffer.getvalue(), "image/png")
        print("Copied figure to clipboard as PNG")
