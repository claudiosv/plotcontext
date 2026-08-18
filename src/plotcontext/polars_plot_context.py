# ruff:file-ignore[too-many-statements-in-try-clause]
import datetime
import inspect
import io
import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict

import clipin
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import polars as pl
import scienceplots
import seaborn as sns
from matplotlib import rcParams, ticker
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from matplotlib.typing import ColorType, LineStyleType  # ruff:ignore[unused-import]

from plotcontext.plot_context import AbstractPlotContext

if TYPE_CHECKING:
    # This tricks type checkers into giving perfect autocomplete for ctx.sns and ctx.plt
    import matplotlib.pyplot as plt_module
    import seaborn as sns_module

science_stylesheets = scienceplots.stylesheets
# plt.style.core.available[:] = sorted(plt.style.library.keys())


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


class SketchConfig(TypedDict, total=False):
    """Configuration dictionary for generation parameters."""

    scale: float = 1.0
    length: float = 100
    randomness: float = 2


class ModuleProxy:
    """Proxies module calls to dynamically inject PlotContext parameters (like ax)."""

    def __init__(self, module: Any, context: PlotContext):
        self._module = module
        self._context = context

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._context._inject_and_call(attr, *args, **kwargs)

            return wrapper
        return attr


class HistogramStat(StrEnum):
    """
    Statistical normalization methods for histogram binning.

    Attributes
    ----------
    COUNT : str
        Show the number of observations in each bin.
    FREQUENCY : str
        Show the number of observations divided by the bin width.
    PROBABILITY : str
        Normalize such that bar heights sum to 1.
    PROPORTION : str
        Normalize such that bar heights sum to 1 (alias/equivalent to PROBABILITY).
    PERCENT : str
        Normalize such that bar heights sum to 100.
    DENSITY : str
        Normalize such that the total area of the histogram equals 1.
    """

    COUNT = "count"
    FREQUENCY = "frequency"
    PROBABILITY = "probability"
    PROPORTION = "proportion"
    PERCENT = "percent"
    DENSITY = "density"


class TickFormatters:
    """A collection of aliases pointing directly to Matplotlib tick formatter classes."""

    NullFormatter: type[ticker.NullFormatter] = ticker.NullFormatter
    """No labels on the ticks."""

    FixedFormatter: type[ticker.FixedFormatter] = ticker.FixedFormatter
    """Set the strings manually for the labels."""

    FuncFormatter: type[ticker.FuncFormatter] = ticker.FuncFormatter
    """User defined function sets the labels."""

    StrMethodFormatter: type[ticker.StrMethodFormatter] = ticker.StrMethodFormatter
    """Use string format method."""

    FormatStrFormatter: type[ticker.FormatStrFormatter] = ticker.FormatStrFormatter
    """Use an old-style sprintf format string."""

    ScalarFormatter: type[ticker.ScalarFormatter] = ticker.ScalarFormatter
    """Default formatter for scalars: autopick the format string."""

    LogFormatter: type[ticker.LogFormatter] = ticker.LogFormatter
    """Formatter for log axes."""

    LogFormatterExponent: type[ticker.LogFormatterExponent] = (
        ticker.LogFormatterExponent
    )
    """Format values for log axis using exponent = log_base(value)."""

    LogFormatterMathtext: type[ticker.LogFormatterMathtext] = (
        ticker.LogFormatterMathtext
    )
    """Format values for log axis using exponent = log_base(value) using Math text."""

    LogFormatterSciNotation: type[ticker.LogFormatterSciNotation] = (
        ticker.LogFormatterSciNotation
    )
    """Format values for log axis using scientific notation."""

    LogitFormatter: type[ticker.LogitFormatter] = ticker.LogitFormatter
    """Probability formatter."""

    EngFormatter: type[ticker.EngFormatter] = ticker.EngFormatter
    """Format labels in engineering notation."""

    PercentFormatter: type[ticker.PercentFormatter] = ticker.PercentFormatter
    """Format labels as a percentage."""


class SinglePlotContext(AbstractPlotContext):
    """
    Class-based context manager for setting up a Seaborn-styled Matplotlib plot.

    Update matplotlib rcParams for legibility in Google Slides.

    Optimizes figure size for the standard 16:9 widescreen layout
    (10 x 5.625 inches) and scales font sizes appropriately for
    presentation viewing distances.
    """

    @staticmethod
    def slides_rc_params() -> dict[mpl.typing.RcKeyType, Any]:
        return {
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

    def __init__(
        self,
        *,
        # title: str | dict[str, Any] | None = None,
        # x_label: str | None = None,
        # y_label: str | None = None,
        # fig_x: float | None = None,
        # fig_y: float | None = None,
        fig_scale: float = 1.0,
        # font_size: int | None = None,
        # title_size: int | None = None,
        # despine: bool = False,
        figsize: tuple[float, float] | None = None,
        sketch: bool | SketchConfig = False,
        # legend: bool | dict[str, Any] = False,
        # legend_outside: bool = False,
        # legend_top: bool = False,
        # grid: Literal["both", "x", "y"] | bool | None = None,
        style: dict[str, Any]
        | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
        styles: list[str] | None = None,
        # ticks: bool | None = None,
        dpi: int | None = 300,
        color_palette: str = "colorblind",
        color_map: dict[Any, int] | None = None,
        font_scale: float = 1.0,
        plot_context: Literal[
            "paper", "notebook", "talk", "poster", "ieee"
        ] = "notebook",
        tight_layout: bool = False,
        # log_x: bool = False,
        # log_y: bool = False,
        # formatter_x: Any | None = None,
        # formatter_y: Any | None = None,
        rc_params: dict[mpl.typing.RcKeyType, Any] | None = None,
        debug: bool = False,
        create_fig: bool = True,
    ) -> None:
        # self.title = title
        # self.label_x = x_label
        # self.label_y = y_label

        # Copy this so sketch defaults never mutate the caller's dictionary.
        self.rc_params: dict[mpl.typing.RcKeyType, Any] = dict(rc_params or {})

        rc_figx, rc_figy = self.rc_params.get("figure.figsize") or rcParams.get(
            "figure.figsize", (10.0, 5.625)
        )

        self.fig_x = figsize[0] or rc_figx
        self.fig_y = figsize[1] or rc_figy
        self.fig_scale = fig_scale
        # self.font_size = font_size
        # self.title_size = title_size
        # self.despine = despine
        self.sketch = sketch
        # self.legend = legend
        # self.legend_outside = legend_outside
        # self.legend_top = legend_top
        self.style = style
        self.styles = styles
        # self.grid = grid
        # self.ticks = ticks
        # print(f"dpi: {dpi}, rcParams dpi: {rcParams.get('figure.dpi', 'not set')}")
        self.dpi = dpi or self.rc_params.get("figure.dpi", 300)
        self.font_scale = font_scale
        self.tight_layout = tight_layout
        self.plot_context = plot_context
        # self.log_x = log_x
        # self.log_y = log_y

        # State tracking
        self.figure: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.exited = False
        self._drawn = False
        self._stack = ExitStack()

        # Palettes and Color Maps
        self.color_palette = color_palette
        self._palette = sns.color_palette(self.color_palette)
        self.color_map = (
            {k: self._palette[v] for k, v in color_map.items()} if color_map else None
        )
        self._drawn = False
        # self.formatter_x = formatter_x
        # self.formatter_y = formatter_y
        if self.sketch:
            self.rc_params.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["xkcd Script", "Comic Sans MS", "Arial"],
                "font.serif": ["xkcd Script", "Comic Sans MS", "Arial"],
            })

        self.debug = debug
        self.create_fig = create_fig

        # if self.debug:
        #     print(
        #         f"Styles: {sorted(plt.style.library.keys())}\n"
        #         f"Available: {sorted(plt.style.available)}\n"
        #         f"Science: {science_stylesheets.keys()}"
        #     )

    @property
    def sns(self) -> sns_module:
        """Proxy for seaborn module with automatic kwarg injection and perfect type hints."""
        return ModuleProxy(sns, self)  # type: ignore

    @property
    def plt(self) -> plt_module:
        """Proxy for pyplot module with automatic kwarg injection and perfect type hints."""
        return ModuleProxy(plt, self)  # type: ignore

    def set_title(
        self,
        label: str,
        fontdict: dict[str, Any] | None = None,
        loc: Literal["center", "left", "right"] | None = None,
        pad: float | None = None,
        *,
        y: float | None = None,
        **kwargs: Any,
    ) -> mtext.Text:
        """
        Set a title for the axes.

        Parameters
        ----------
        label : str
            The text to use for the title.
        fontdict : dict[str, Any] | None, default: None
            A dictionary controlling the appearance of the title text.
            Overrides default styling from rcParams.
        loc : Literal["center", "left", "right"] | None, default: None
            Which title to set and its horizontal alignment. If None,
            defaults to rcParams["axes.titlelocation"].
        pad : float | None, default: None
            The offset of the title from the top of the axes, in points.
            If None, defaults to rcParams["axes.titlepad"].
        y : float | None, default: None
            Vertical axes location for the title (1.0 is exactly on the top edge).
            If None, defaults to rcParams["axes.titley"] or an auto-calculated
            position avoiding overlaps.
        **kwargs : Any
            Additional keyword arguments are text properties passed directly
            to the underlying Matplotlib Text instance (e.g., fontsize=14,
            color='red', fontweight='bold', zorder=10).

        Returns
        -------
        matplotlib.text.Text
            The Matplotlib Text instance representing the drawn title.
        """
        return self.ax.set_title(
            label=label,
            fontdict=fontdict,
            loc=loc,
            pad=pad,
            y=y,
            **kwargs,
        )

    def _activate(self) -> None:
        if self.figure is None or self.ax is None:
            msg = "PlotContext has no active figure"
            raise RuntimeError(msg)

        if not plt.fignum_exists(self.figure.number):
            msg = "The PlotContext figure has already been closed"
            raise RuntimeError(msg)

        plt.figure(self.figure.number)
        plt.sca(self.ax)

    def __enter__(self) -> tuple[Self, plt.Figure, plt.Axes]:
        try:
            # plt.ion() is a context manager, so this restores the previous
            # interactive-mode state when the ExitStack is closed.
            # self._stack.enter_context(plt.ion())
            self._setup_styles()

            if self.create_fig:
                figsize = (
                    self.fig_x * self.fig_scale,
                    self.fig_y * self.fig_scale,
                )

                self.figure, self.ax = plt.subplots(
                    figsize=figsize,
                    dpi=self.dpi,
                )
                self._activate()
        except BaseException:
            # Python does not invoke __exit__ when __enter__ fails.
            self._stack.close()
            raise
        return self, self.figure, self.ax

    def _setup_styles(self) -> None:
        """Manage all context layers via ExitStack to ensure clean teardown."""
        contexts: list[AbstractContextManager] = []

        if self.plot_context == "ieee":
            contexts.append(plt.style.context(["science", "ieee"]))
        else:
            # Unlike sns.set_theme(), all of these changes are scoped and will
            # be restored by the ExitStack.
            contexts.extend([
                sns.plotting_context(
                    context=self.plot_context,
                    font_scale=self.font_scale,
                    rc=self.rc_params,
                ),
                sns.axes_style(self.style),
                self._palette,
            ])

        if self.styles:
            contexts.append(plt.style.context(self.styles))

        if self.sketch:
            xkcd_kwargs = self.sketch if isinstance(self.sketch, dict) else {}
            contexts.append(plt.xkcd(**xkcd_kwargs))

        # Enter this last so explicit caller overrides take precedence.
        contexts.append(plt.rc_context(self.rc_params))

        for ctx in contexts:
            self._stack.enter_context(ctx)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Literal[False]:
        try:
            if exc_type is not None:
                # Do not draw or show a partially completed plot. Returning
                # False preserves the original exception and traceback.
                print(f"PlotContext body raised {exc_type.__name__}: {exc_value}")
            else:
                if self.ax is None:
                    warnings.warn("No ax found on exit", stacklevel=2)
                else:
                    self._draw()

                # Show exactly once, and only after successful body execution.
                plt.show()
        finally:
            # Always release the owned figure and all scoped state, including
            # when _draw(), show(), or close() raises.
            try:
                self._close()
            finally:
                self._stack.close()
                self.exited = True

        return False

    def _accepts_kwarg(self, func: Callable[..., Any], name: str) -> bool:
        func_name = getattr(func, "__name__", repr(func))
        no_ax = {
            # "histplot", "kdeplot", "displot",
            "relplot",
            "catplot",
            "move_legend",
            "FacetGrid",
        }

        if name == "ax" and func_name in no_ax:
            return False

        try:
            sig = inspect.signature(func)
            name_in_sig = name in sig.parameters

            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            accepts = name_in_sig or has_kwargs
            if self.debug:
                print(
                    f"Function {func_name} accepts '{name}': {accepts} "
                    f"(name_in_sig={name_in_sig}, keyword_in_params={has_kwargs})"
                )
                if has_kwargs:
                    print(f"params: {sig.parameters}")
                    var_params = [
                        p.name
                        for p in sig.parameters.values()
                        if p.kind == inspect.Parameter.VAR_KEYWORD
                    ]
                    print(f"var params: {var_params}")
        except ValueError:
            return False
        else:
            return accepts

    def _inject_and_call(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if self._accepts_kwarg(func, "ax") and kwargs.get("ax") is None:
            if self.ax is None:
                self.ax = plt.gca()
            kwargs["ax"] = self.ax

        if (
            self.color_map
            and self._accepts_kwarg(func, "palette")
            and kwargs.get("palette") is None
        ):
            kwargs["palette"] = self.color_map

        if self.figure is not None and self.ax is not None:
            self._activate()

        result = func(*args, **kwargs)

        if self.debug:
            kwargs.pop("data", None)  # Avoid dumping large DataFrames in debug logs
            print(
                f"Called {func.__name__} with args={args} kwargs={kwargs},\n"
                f"Result: {result} type={type(result).__name__}"
            )

        if hasattr(result, "axes"):
            if result.axes != self.ax:
                print("Warning: access don't match:", result.axes, self.ax)

            self.ax = result.axes
        elif isinstance(result, Axes):
            if result != self.ax:
                print("Warning: access don't match:", result, self.ax)

            self.ax = result
        # elif hasattr(result, "ax") and isinstance(result.ax, Axes):
        #     self.ax = result.ax
        # elif hasattr(result, "axes") and isinstance(result.axes, Axes):
        #     self.ax = result.axes

        return result

    def save(self, filename: str, **save_kwargs) -> None:
        """Save the current figure with the given filename and additional savefig kwargs."""
        if self.figure is None:
            msg = "No figure to save. Ensure you're within the context."
            raise RuntimeError(msg)

        # Default savefig kwargs for publication quality
        default_kwargs = {
            "format": "pdf",
            # "bbox_inches": "tight",
            # "pad_inches": 0.01,
            # "dpi": self.dpi,
        }
        default_kwargs.update(save_kwargs)

        if not self._drawn:
            if self.ax is None:
                msg = "No axes to finalize. Ensure you're within the context."
                raise RuntimeError(msg)
            self._draw()
        self.figure.savefig(filename, **default_kwargs)
        print(f"Saved figure to {filename} with kwargs: {default_kwargs}")

    def _close(self) -> None:
        if self.figure is not None:
            # Never close figures that this context does not own.
            plt.close(self.figure)

    def _draw(self) -> None:
        # if self.despine:
        #     sns.despine(left=True)

        # if self.log_x:
        #     self.ax.set(xscale="log")  # , xticks=[15, 30, 59, 118, 236, 470])
        #     formatter = self.formatter_x or ScalarFormatter()
        #     self.ax.xaxis.set_major_formatter(formatter)
        # if self.log_y:
        #     self.ax.set(yscale="log")  # , yticks=[15, 30, 59, 118, 236, 470])
        #     formatter = self.formatter_y or ScalarFormatter()
        #     self.ax.yaxis.set_major_formatter(formatter)

        # if self.legend:
        #     if self.legend_outside and self.legend_top:
        #         msg = "Cannot have legend_outside and legend_top both True"
        #         raise ValueError(msg)

        #     legend_kwargs = self.legend if isinstance(self.legend, dict) else {}
        #     xlegend = self.ax.legend(**legend_kwargs)

        #     if self.legend_top:
        #         xlegend.set_bbox_to_anchor((0.5, 1.15))
        #         # xlegend._loc = 9  # 'upper center'
        #         xlegend.set_loc(9)
        #     elif self.legend_outside:
        #         xlegend.set_bbox_to_anchor((1.05, 1.0))
        #         xlegend.borderaxespad = 0.0

        if self.ax.get_legend() and self.sketch:
            for text in self.ax.get_legend().get_texts():
                text.set_fontfamily("xkcd Script")
            # if self.sketch:
            #     for text in xlegend.get_texts():
            #         text.set_fontfamily("xkcd Script")

        # if self.grid:
        #     grid_axis = "both" if self.grid is True else self.grid
        #     self.ax.grid(
        #         visible=True,
        #         linestyle="dashed",
        #         alpha=0.6,
        #         axis=grid_axis,
        #     )

        if self.tight_layout:
            plt.tight_layout()

        # plt.show()
        self._drawn = True

    def save_fig(self, output_path: Path | str, name: str) -> None:
        output_path = Path(output_path)
        if not isinstance(name, str):
            msg = f"Expected name to be a string, got {type(name).__name__}."
            raise TypeError(msg)

        name = name.strip().replace("/", "_")
        if output_path.exists() and not output_path.is_dir():
            msg = f"Output path {output_path} exists and is not a directory"
            raise ValueError(msg)

        output_path.mkdir(parents=True, exist_ok=True)
        today_date = datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )

        pdf_filename = output_path / f"{name}_{today_date}.pdf"
        png_filename = output_path / f"{name}_{today_date}.png"

        target_fig = self.figure  # or plt.gcf()

        if not self.exited:
            warnings.warn(
                "Save_fig invoked before exit, figure might be incomplete", stacklevel=2
            )

        target_fig.savefig(pdf_filename)
        target_fig.savefig(png_filename, transparent=True)


class PlotContext(AbstractPlotContext):
    """
    Class-based context manager for setting up a Seaborn-styled Matplotlib plot.

    Update matplotlib rcParams for legibility in Google Slides.

    Optimizes figure size for the standard 16:9 widescreen layout
    (10 x 5.625 inches) and scales font sizes appropriately for
    presentation viewing distances.
    """

    @staticmethod
    def slides_rc_params() -> dict[str, Any]:
        return {
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

    def __init__(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        fig_x: float | None = None,
        fig_y: float | None = None,
        fig_scale: float = 1.0,
        font_size: int | None = None,
        title_size: int | None = None,
        despine: bool = False,
        sketch: bool | SketchConfig = False,
        legend: bool | dict[str, Any] = False,
        legend_outside: bool = False,
        legend_top: bool = False,
        grid: Literal["both", "x", "y"] | bool | None = None,
        style: dict[str, Any]
        | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
        styles: list[str] | None = None,
        ticks: bool | None = None,
        dpi: int | None = 300,
        color_palette: str = "colorblind",
        color_map: dict[Any, int] | None = None,
        font_scale: float = 1.0,
        plot_context: Literal[
            "paper", "notebook", "talk", "poster", "ieee"
        ] = "notebook",
        tight_layout: bool = False,
        log_x: bool = False,
        log_y: bool = False,
        formatter_x: Any | None = None,
        formatter_y: Any | None = None,
        rc_params: dict[str, Any] | None = None,
        debug: bool = False,
        create_fig: bool = True,
    ) -> None:
        self.title = title
        self.label_x = x_label
        self.label_y = y_label

        # Copy this so sketch defaults never mutate the caller's dictionary.
        self.rc_params = dict(rc_params or {})

        rc_figx, rc_figy = self.rc_params.get("figure.figsize") or rcParams.get(
            "figure.figsize", (10.0, 5.625)
        )

        self.fig_x = fig_x or rc_figx
        self.fig_y = fig_y or rc_figy
        self.fig_scale = fig_scale
        self.font_size = font_size
        self.title_size = title_size
        self.despine = despine
        self.sketch = sketch
        self.legend = legend
        self.legend_outside = legend_outside
        self.legend_top = legend_top
        self.style = style
        self.styles = styles
        self.grid = grid
        self.ticks = ticks
        # print(f"dpi: {dpi}, rcParams dpi: {rcParams.get('figure.dpi', 'not set')}")
        self.dpi = dpi or self.rc_params.get("figure.dpi", 300)
        self.font_scale = font_scale
        self.tight_layout = tight_layout
        self.plot_context = plot_context
        self.log_x = log_x
        self.log_y = log_y

        # State tracking
        self.figure: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.exited = False
        self._drawn = False
        self._stack = ExitStack()

        # Palettes and Color Maps
        self.color_palette = color_palette
        self._palette = sns.color_palette(self.color_palette)
        self.color_map = (
            {k: self._palette[v] for k, v in color_map.items()} if color_map else None
        )
        self._drawn = False
        self.formatter_x = formatter_x
        self.formatter_y = formatter_y
        if self.sketch:
            self.rc_params.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["xkcd Script", "Comic Sans MS", "Arial"],
                "font.serif": ["xkcd Script", "Comic Sans MS", "Arial"],
            })

        self.debug = debug
        self.create_fig = create_fig

    @property
    def sns(self) -> sns_module:
        """Proxy for seaborn module with automatic kwarg injection and perfect type hints."""
        return ModuleProxy(sns, self)  # type: ignore

    @property
    def plt(self) -> plt_module:
        """Proxy for pyplot module with automatic kwarg injection and perfect type hints."""
        return ModuleProxy(plt, self)  # type: ignore

    def __enter__(self) -> Self:
        try:
            # plt.ion() is a context manager, so this restores the previous
            # interactive-mode state when the ExitStack is closed.
            # self._stack.enter_context(plt.ion())
            self._setup_styles()

            if self.create_fig:
                self.figure, self.ax = plt.subplots(
                    figsize=(self.fig_x * self.fig_scale, self.fig_y * self.fig_scale),
                )
        except BaseException:
            # Python does not invoke __exit__ when __enter__ fails.
            self._stack.close()
            raise
        return self

    def _setup_styles(self) -> None:
        """Manage all context layers via ExitStack to ensure clean teardown."""
        contexts: list[AbstractContextManager] = []

        if self.plot_context == "ieee":
            contexts.append(plt.style.context(["science", "ieee"]))
        else:
            # Unlike sns.set_theme(), all of these changes are scoped and will
            # be restored by the ExitStack.
            contexts.extend([
                sns.plotting_context(
                    context=self.plot_context,
                    font_scale=self.font_scale,
                ),
                sns.axes_style(self.style),
                self._palette,
            ])

        if self.styles:
            contexts.append(plt.style.context(self.styles))

        if self.sketch:
            xkcd_kwargs = self.sketch if isinstance(self.sketch, dict) else {}
            contexts.append(plt.xkcd(**xkcd_kwargs))

        # Enter this last so explicit caller overrides take precedence.
        contexts.append(plt.rc_context(self.rc_params))

        for ctx in contexts:
            self._stack.enter_context(ctx)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Literal[False]:
        try:
            if exc_type is not None:
                # Do not draw or show a partially completed plot. Returning
                # False preserves the original exception and traceback.
                print(f"PlotContext body raised {exc_type.__name__}: {exc_value}")
            else:
                if self.ax is None:
                    warnings.warn("No ax found on exit", stacklevel=2)
                else:
                    self._draw()

                # Show exactly once, and only after successful body execution.
                plt.show()
        finally:
            # Always release the owned figure and all scoped state, including
            # when _draw(), show(), or close() raises.
            try:
                self._close()
            finally:
                self._stack.close()
                self.exited = True

        return False

    # @staticmethod
    def _accepts_kwarg(self, func: Callable[..., Any], name: str) -> bool:
        func_name = getattr(func, "__name__", repr(func))
        no_ax = {
            # "histplot", "kdeplot", "displot",
            "relplot",
            "catplot",
            "move_legend",
            "FacetGrid",
        }

        if name == "ax" and func_name in no_ax:
            return False

        try:
            sig = inspect.signature(func)
            name_in_sig = name in sig.parameters

            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            accepts = name_in_sig or has_kwargs
            if self.debug:
                print(
                    f"Function {func_name} accepts '{name}': {accepts} "
                    f"(name_in_sig={name_in_sig}, keyword_in_params={has_kwargs})"
                )
                if has_kwargs:
                    print(f"params: {sig.parameters}")
                    var_params = [
                        p.name
                        for p in sig.parameters.values()
                        if p.kind == inspect.Parameter.VAR_KEYWORD
                    ]
                    print(f"var params: {var_params}")
        except ValueError:
            return False
        else:
            return accepts

    def _inject_and_call(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        # if self.ax is None:
        #     self.ax = plt.gca()

        if self._accepts_kwarg(func, "ax") and kwargs.get("ax") is None:
            if self.ax is None:
                self.ax = plt.gca()
            kwargs["ax"] = self.ax

        if (
            self.color_map
            and self._accepts_kwarg(func, "palette")
            and kwargs.get("palette") is None
        ):
            kwargs["palette"] = self.color_map

        annotate = kwargs.pop("annotate", False)
        if annotate:
            if annotate not in {"x", "y"}:
                msg = "Must specify which axis to annotate ('x' or 'y')"
                raise ValueError(msg)

            if func.__name__ in {"boxplot", "violinplot"}:
                return self._apply_boxplot_annotations(func, annotate, *args, **kwargs)
            msg = f"Magic annotate not supported for {func.__name__}"
            raise ValueError(msg)

        result = func(*args, **kwargs)

        if self.debug:
            kwargs.pop("data", None)  # Avoid dumping large DataFrames in debug logs
            print(
                f"Called {func.__name__} with args={args} kwargs={kwargs},\n"
                f"Result: {result} type={type(result).__name__}"
            )

        if hasattr(result, "axes"):
            self.ax = result.axes
        elif isinstance(result, Axes):
            self.ax = result
        # elif hasattr(result, "ax") and isinstance(result.ax, Axes):
        #     self.ax = result.ax
        # elif hasattr(result, "axes") and isinstance(result.axes, Axes):
        #     self.ax = result.axes

        return result

    def save(self, filename: str, **save_kwargs) -> None:
        """Save the current figure with the given filename and additional savefig kwargs."""
        if self.figure is None:
            msg = "No figure to save. Ensure you're within the context."
            raise RuntimeError(msg)

        # Default savefig kwargs for publication quality
        default_kwargs = {
            "format": "pdf",
            # "bbox_inches": "tight",
            # "pad_inches": 0.01,
            # "dpi": self.dpi,
        }
        default_kwargs.update(save_kwargs)

        if not self._drawn:
            if self.ax is None:
                msg = "No axes to finalize. Ensure you're within the context."
                raise RuntimeError(msg)
            self._draw()
        self.figure.savefig(filename, **default_kwargs)
        print(f"Saved figure to {filename} with kwargs: {default_kwargs}")

    def clip(self, **save_kwargs):
        """Save the current figure with the given filename and additional savefig kwargs."""
        if self.figure is None:
            msg = "No figure to save. Ensure you're within the context."
            raise RuntimeError(msg)

        # Default savefig kwargs for publication quality
        default_kwargs = {
            "format": "png",
            "dpi": 300,
            "bbox_inches": "tight",
            "transparent": True,
        }
        default_kwargs.update(save_kwargs)

        if not self._drawn:
            self._draw()
        # self.figure.savefig(filename, **default_kwargs)
        # print(f"Saved figure to {filename} with kwargs: {default_kwargs}")

        with io.BytesIO() as buffer:
            # Save figure to buffer instead of a file on your hard drive
            # Using high DPI and tight bbox ensures it looks great in Slides
            plt.savefig(buffer, **default_kwargs)

            # Reset the buffer's position to the beginning before reading
            buffer.seek(0)

            # Extract the raw image bytes
            image_bytes = buffer.getvalue()
        clipin.copy(image_bytes, "image/png")

        # 3. Push the bytes to the macOS clipboard
        # import pasteboard
        # pb = pasteboard.Pasteboard()
        # pb.set_contents(image_bytes, pasteboard.PNG)
        # plt.savefig('slide_plot.png', dpi=300, transparent=True, bbox_inches='tight')

    @staticmethod
    def _apply_boxplot_annotations(
        func: Callable[..., Any], annotate: str, *args: Any, **kwargs: Any
    ) -> tuple[Axes, pl.DataFrame]:
        orig_data: pl.DataFrame = kwargs["data"]
        hue = kwargs.get("hue")
        y = kwargs.get(annotate)
        x = kwargs.get("x" if annotate == "y" else "y")
        x_align = kwargs.get("x_align", "median")

        # Get unique values while preserving order of appearance if not explicitly provided
        hue_order = (
            kwargs.get(
                "hue_order",
                orig_data.get_column(hue).unique(maintain_order=True).to_list(),
            )
            if hue
            else None
        )
        if hue:
            kwargs["hue_order"] = hue_order

        order = kwargs.get(
            "order", orig_data.get_column(y).unique(maintain_order=True).to_list()
        )
        kwargs["order"] = order

        data = orig_data.clone()
        group = []

        # Cast grouping variables to pl.Enum to enforce deterministic ordering in the groupby
        data = data.with_columns(pl.col(y).cast(pl.Enum(order)))
        group.append(y)

        if hue and hue_order is not None and hue != y:
            data = data.with_columns(pl.col(hue).cast(pl.Enum(hue_order)))
            group.append(hue)

        # Determine aggregate expressions based on column presence
        if "count" in data.columns:
            agg_exprs = [
                pl.col("count").sum().alias("n"),
                getattr(pl.col(x), x_align)().alias("x_align"),
            ]
            print("Using existing 'count' column for annotations")
        else:
            agg_exprs = [
                pl.len().alias("n"),
                getattr(pl.col(x), x_align)().alias("x_align"),
            ]

        # Group, aggregate, and strictly sort by the Enums to match plot rendering order
        counts = data.group_by(group).agg(agg_exprs).sort(group)

        kwargs["data"] = data
        g: Axes = func(*args, **kwargs)

        box_spacing = 1  # / len(hue_order) if hue_order else 1

        # iter_rows(named=True) is incredibly fast and yields standard Python dicts
        for i, row in enumerate(counts.iter_rows(named=True)):
            n = int(row["n"])
            offset = i * box_spacing

            x_loc = row["x_align"] + 0.5 if annotate == "y" else offset
            y_loc = offset if annotate == "y" else row["x_align"] * 1.025
            print(f"Annotating group {i} at ({x_loc}, {y_loc}) with n={n}")
            g.text(
                x_loc,
                y_loc,
                f"n={n:,}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.5,
                },
            )

            # Keep tracking history purely for logic continuity (matching old implementation)
            tuple(row[g_col] for g_col in group)

        return g, counts

    def _close(self) -> None:
        if self.figure is not None:
            # Never close figures that this context does not own.
            plt.close(self.figure)

    def _draw(self) -> None:
        if self.label_x:
            self.ax.set_xlabel(self.label_x)  # , fontsize=self.font_size)
        if self.label_y:
            self.ax.set_ylabel(self.label_y)  # , fontsize=self.font_size)

        if self.title:
            if isinstance(self.title, dict):
                self.ax.set_title(**self.title)
            else:
                self.ax.set_title(self.title)  # , fontsize=self.title_size)

        if self.ticks:
            self.ax.tick_params(
                axis="x"
            )  # , labelsize=plt.rcParams["xtick.labelsize"])
            self.ax.tick_params(
                axis="y"
            )  # , labelsize=plt.rcParams["ytick.labelsize"])

        if self.despine:
            sns.despine(left=True)

        if self.log_x:
            self.ax.set(xscale="log")  # , xticks=[15, 30, 59, 118, 236, 470])
            formatter = self.formatter_x or ScalarFormatter()
            self.ax.xaxis.set_major_formatter(formatter)
        if self.log_y:
            self.ax.set(yscale="log")  # , yticks=[15, 30, 59, 118, 236, 470])
            formatter = self.formatter_y or ScalarFormatter()
            self.ax.yaxis.set_major_formatter(formatter)

        if self.legend:
            if self.legend_outside and self.legend_top:
                msg = "Cannot have legend_outside and legend_top both True"
                raise ValueError(msg)

            legend_kwargs = self.legend if isinstance(self.legend, dict) else {}
            xlegend = self.ax.legend(**legend_kwargs)

            if self.legend_top:
                xlegend.set_bbox_to_anchor((0.5, 1.15))
                # xlegend._loc = 9  # 'upper center'
                xlegend.set_loc(9)
            elif self.legend_outside:
                xlegend.set_bbox_to_anchor((1.05, 1.0))
                xlegend.borderaxespad = 0.0

            if self.sketch:
                for text in xlegend.get_texts():
                    text.set_fontfamily("xkcd Script")

        if self.grid:
            grid_axis = "both" if self.grid is True else self.grid
            self.ax.grid(
                visible=True,
                linestyle="dashed",
                alpha=0.6,
                axis=grid_axis,
            )

        if self.tight_layout:
            plt.tight_layout()

        # plt.show()
        self._drawn = True

    def save_fig(self, output_path: Path | str, name: str) -> None:
        output_path = Path(output_path)
        if not isinstance(name, str):
            msg = f"Expected name to be a string, got {type(name).__name__}."
            raise TypeError(msg)

        name = name.strip().replace("/", "_")
        if output_path.exists() and not output_path.is_dir():
            msg = f"Output path {output_path} exists and is not a directory"
            raise ValueError(msg)

        output_path.mkdir(parents=True, exist_ok=True)
        today_date = datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )

        pdf_filename = output_path / f"{name}_{today_date}.pdf"
        png_filename = output_path / f"{name}_{today_date}.png"

        target_fig = self.figure  # or plt.gcf()

        if not self.exited:
            warnings.warn(
                "Save_fig invoked before exit, figure might be incomplete", stacklevel=2
            )

        target_fig.savefig(pdf_filename)
        target_fig.savefig(png_filename, transparent=True)
