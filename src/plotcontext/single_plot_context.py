import datetime
import inspect
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import scienceplots  # noqa: F401 - registers "science"/"ieee" plt.style entries
import seaborn as sns
from matplotlib import rcParams
from matplotlib.axes import Axes

from plotcontext.plot_context import AbstractPlotContext
from plotcontext.polars_plot_context import ModuleProxy, SketchConfig

if TYPE_CHECKING:
    # This tricks type checkers into giving perfect autocomplete for ctx.sns and ctx.plt
    import matplotlib.pyplot as plt_module
    import seaborn as sns_module


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
        fig_scale: float = 1.0,
        figsize: tuple[float, float] | None = None,
        sketch: bool | SketchConfig = False,
        style: dict[str, Any]
        | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
        styles: list[str] | None = None,
        dpi: int | None = 300,
        color_palette: str = "colorblind",
        color_map: dict[Any, int] | None = None,
        font_scale: float = 1.0,
        plot_context: Literal[
            "paper", "notebook", "talk", "poster", "ieee"
        ] = "notebook",
        tight_layout: bool = False,
        rc_params: dict[mpl.typing.RcKeyType, Any] | None = None,
        debug: bool = False,
        create_fig: bool = True,
    ) -> None:
        # Copy this so sketch defaults never mutate the caller's dictionary.
        self.rc_params: dict[mpl.typing.RcKeyType, Any] = dict(rc_params or {})

        rc_figx, rc_figy = self.rc_params.get("figure.figsize") or rcParams.get(
            "figure.figsize", (10.0, 5.625)
        )

        self.fig_x = figsize[0] or rc_figx
        self.fig_y = figsize[1] or rc_figy
        self.fig_scale = fig_scale
        self.sketch = sketch
        self.style = style
        self.styles = styles
        self.dpi = dpi or self.rc_params.get("figure.dpi", 300)
        self.font_scale = font_scale
        self.tight_layout = tight_layout
        self.plot_context = plot_context

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
        if self.sketch:
            self.rc_params.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["xkcd Script", "Comic Sans MS", "Arial"],
                    "font.serif": ["xkcd Script", "Comic Sans MS", "Arial"],
                }
            )

        self.debug = debug
        self.create_fig = create_fig

    @property
    def sns(self) -> sns_module:
        """Proxy for seaborn with automatic kwarg injection and perfect type hints."""
        return ModuleProxy(sns, self)  # type: ignore

    @property
    def plt(self) -> plt_module:
        """Proxy for pyplot with automatic kwarg injection and perfect type hints."""
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
            contexts.extend(
                [
                    sns.plotting_context(
                        context=self.plot_context,
                        font_scale=self.font_scale,
                        rc=self.rc_params,
                    ),
                    sns.axes_style(self.style),
                    self._palette,
                ]
            )

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

        return result

    def save(self, filename: str, **save_kwargs) -> None:
        """Save the current figure to filename with the given savefig kwargs."""
        if self.figure is None:
            msg = "No figure to save. Ensure you're within the context."
            raise RuntimeError(msg)

        # Default savefig kwargs for publication quality
        default_kwargs = {"format": "pdf"}
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
        if self.ax.get_legend() and self.sketch:
            for text in self.ax.get_legend().get_texts():
                text.set_fontfamily("xkcd Script")

        if self.tight_layout:
            plt.tight_layout()

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

        target_fig = self.figure

        if not self.exited:
            warnings.warn(
                "Save_fig invoked before exit, figure might be incomplete", stacklevel=2
            )

        target_fig.savefig(pdf_filename)
        target_fig.savefig(png_filename, transparent=True)
