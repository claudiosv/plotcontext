import datetime
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import matplotlib as mpl
import pandas as pd

# import scienceplots  # noqa: F401
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401
from pandas import DataFrame
from statsmodels.stats.descriptivestats import is_categorical_dtype

if TYPE_CHECKING:
    import types
    from collections.abc import Callable


def set_icml_style():
    msg = "Deprecated sicne it affects global state. Use PlotContext(plot_context='ieee') instead."
    raise ValueError(msg)
    icml_params = get_icml_style()
    plt.rcParams.update(icml_params)


def get_icml_style():
    # ICML column width is 3.25 inches
    width = 3.25
    # Golden ratio (0.618) is a good default height
    height = width * 1  # 0.618

    return {
        "axes.labelsize": 8,
        "axes.titlepad": 0,
        "axes.titlesize": 9,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.hspace": 0.01,
        "figure.constrained_layout.use": True,  # Global toggle
        "figure.constrained_layout.w_pad": 0.02,
        "figure.dpi": 300,
        "figure.figsize": (width, height),
        "font.family": "serif",
        "font.serif": ["Linux Libertine", "Libertine", "DejaVu Serif"],
        "font.size": 9,
        "legend.borderaxespad": 0,
        "legend.borderpad": 0,
        "legend.fontsize": 7,
        "lines.markersize": 3,
        # "pdf.fonttype": 42,
        # "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": True,
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{libertine} \usepackage[libertine]{newtxmath}",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }


@contextmanager
def icml_style(width=3.25, height_ratio=1.0):
    """Context manager for ICML-compliant figures."""
    height = width * height_ratio

    # Define the specific parameters
    icml_params = get_icml_style()
    icml_params.update({"figure.figsize": (width, height)})

    # Use Matplotlib's built-in rc_context to handle the temporary swap
    with plt.rc_context(rc=icml_params):
        yield


def set_rcParams() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,  # High resolution for inline plots
        # %config InlineBackend.figure_format = 'retina' == 200 dpi
        # "figure.figsize": (10, 6),  # Set a larger default figure size
        "lines.antialiased": True,
        # "font.monospace": ["IBM Plex Mono", "monospace"],
        # "font.sans-serif": ["DejaVu Sans", "Open Sans", "sans-serif"],
        # "font.family": "sans-serif",  # "cursive", #"sans-serif", #"serif",  # Use serif font for better readability
        # "font.serif": ["Times New Roman", "DejaVu Serif"],
        #   "text.usetex": True,  # Set to True if using LaTeX for text rendering
        "text.latex.preamble": r"\usepackage{sfmath}",
        # "lines.linewidth": 2,  # Make lines thicker
        # "lines.marker": "o",   # Set a default marker
        # "lines.markersize": 6,
        # "axes.grid": True,  # Enable grid by default
        # "grid.alpha": 0.3,  # Make grid lines less intrusive
    })


class PlotContext:
    """Class-based context manager for setting up a Seaborn-styled Matplotlib plot."""

    ax: plt.Axes
    # _plot_context: _GeneratorContextManager[None, Any, None] | None
    _plot_context: Generator[None, Any, None] | None

    def __init__(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        fig_x: float | None = 16,
        fig_y: float | None = 9,
        fig_scale: float = 1.0,
        font_size: int | None = None,
        title_size: int | None = None,
        despine: bool = False,
        sketch: bool = False,
        legend: bool | dict[str, Any] = False,
        legend_outside: bool = False,
        legend_top: bool = False,
        grid: Literal["both", "x", "y"] | bool | None = None,
        style: dict[str, Any]
        | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
        ticks: bool | None = None,
        dpi: int | None = None,
        color_palette: str = "colorblind",
        color_map: dict[Any, int] | None = None,
        font_scale: float = 1.0,
        plot_context: Literal[
            "paper", "notebook", "talk", "poster", "ieee"
        ] = "notebook",
        tight_layout: bool = False,
    ) -> None:
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.fig_x = fig_x
        self.fig_y = fig_y
        self.fig_scale = fig_scale
        self.font_size = font_size
        self.title_size = title_size
        self.despine = despine
        self.sketch = sketch
        self.legend = legend
        # loc: 'best', 'upper right', 'upper left', 'lower left',
        # 'lower right', 'right', 'center left', 'center right',
        # 'lower center', 'upper center', 'center'
        self.legend_outside = legend_outside
        self.legend_top = legend_top
        self.style = style
        self.grid = grid
        self.ticks = ticks
        self._xkcd_context = None
        self.figure = None
        self.exited = False
        self.dpi = dpi
        self.color_palette = color_palette
        self.font_scale = font_scale
        self.tight_layout = tight_layout
        self._palette = sns.color_palette(color_palette)
        self.color_map = (
            {k: self._palette[v] for k, v in color_map.items()} if color_map else None
        )
        self.plot_context = plot_context
        # if self.plot_context == "ieee":
        #     self._plot_context = plt.style.context(["science", "ieee"])  # , "bright"
        # else:
        self._plot_context = None
        self._icml_style = None

        self._drawn = False
        # if self.dpi is None:
        #     self.dpi = rcParams.get("figure.dpi", 200)
            # print(self.dpi)

    def __enter__(self) -> Self:
        # Disable interactive mode to prevent figures from displaying prematurely
        plt.ion()

        self._enter_contexts()
        # if self.plot_context == "paper":
        #     set_icml_style()
        #     self.fig_x = 3.25  # None
        #     self.fig_y = 3.25  # None

        # if self.fig_x is None or self.fig_y is None:
        #     self.figure = plt.figure()  # dpi=self.dpi)
        # else:
        #     self.figure = plt.figure(
        #         # figsize=(self.fig_x * self.fig_scale, self.fig_y * self.fig_scale),
        #         # dpi=self.dpi,
        #     )

        # self.ax = None
        self.figure, self.ax = plt.subplots(
            figsize=(self.fig_x * self.fig_scale, self.fig_y * self.fig_scale),
        )  # layout="constrained")
        # self.figure.get_layout_engine().set(w_pad=0.05, h_pad=0.05, hspace=0, wspace=0)
        # self.ax = plt.gca()
        # if self.font_size is None:
        #     self.font_size = plt.rcParams["axes.labelsize"]
        # if self.title_size is None:
        #     self.title_size = plt.rcParams["axes.titlesize"]

        return self

    def _enter_contexts(self) -> None:
        if self.plot_context == "ieee":
            # set_icml_style()
            self._plot_context = plt.style.context(["science", "ieee"])
            self._plot_context.__enter__()  # manually enter science style context
            # set_icml_style()
            self._icml_style = icml_style(
                width=self.fig_x, height_ratio=self.fig_y / self.fig_x
            )
            self._icml_style.__enter__()  # manually enter ICML style context
            return

        sns.set_theme(context="notebook", style=self.style, palette="colorblind")
        self._plotting_context = sns.plotting_context(
            context="notebook", font_scale=self.font_scale
        )
        self._plotting_context.__enter__()  # manually enter plotting context

        self._axes_style_context = sns.axes_style(self.style)
        self._axes_style_context.__enter__()  # manually enter axes style context

        self._palette.__enter__()  # manually enter palette context

        # if self.plot_context == "ieee":
        # set_icml_style()
        # print(plt.rcParams)

        if self.sketch:
            self._xkcd_context = plt.xkcd()
            self._xkcd_context.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.ax is None:
            warnings.warn("No ax found on exit", stacklevel=2)
            plt.show()
        else:
            self._draw()

        self._close()

        self._exit_contexts(exc_type, exc_value, traceback)

        self.exited = True

    def _exit_contexts(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:

        if self._plot_context:
            self._plot_context.__exit__(exc_type, exc_value, traceback)
            return

        if self._icml_style:
            self._icml_style.__exit__(exc_type, exc_value, traceback)
            return

        if self._palette:
            self._palette.__exit__(exc_type, exc_value, traceback)

        if self._axes_style_context:
            self._axes_style_context.__exit__(exc_type, exc_value, traceback)

        if self._plotting_context:
            self._plotting_context.__exit__(exc_type, exc_value, traceback)

        if self.sketch and self._xkcd_context:
            self._xkcd_context.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def _inject_kwarg(
        func: Callable[..., T], /, name: str, value: Any, kwargs: dict[str, Any]
    ) -> None:
        if accepts_kwarg(func, name=name) and (
            name not in kwargs or kwargs[name] is None
        ):
            kwargs[name] = value
        return kwargs

    def __call__(
        self, func: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> T | tuple[T, pd.DataFrame]:
        """Call a seaborn or matplotlib function with this context's Axes injected.

        Examples
        --------
        >>> with PlotContext() as pc:
        ...     pc.plot(sns.histplot, data=df, x="col")
        ...     pc.plot(sns.kdeplot, data=df, x="col")
        """
        if self.ax is None:
            self.ax = plt.gca()
        kwargs = self._inject_kwarg(func, "ax", self.ax, kwargs)

        if self.color_map:
            kwargs = self._inject_kwarg(func, "palette", self.color_map, kwargs)

        annotate = kwargs.pop("annotate", False)
        if annotate:
            if annotate not in {"x", "y"}:
                msg = "Must specify which axis to annotate"
                raise ValueError(msg)
            orig_data: DataFrame = kwargs["data"]
            func_name = func.__name__
            match func_name:
                case "boxplot":
                    hue = kwargs.get("hue")
                    hue_order = None
                    if hue is not None:
                        hue_order = kwargs.get("hue_order", orig_data.get(hue).unique())
                        kwargs["hue_order"] = hue_order
                    y = kwargs.get(annotate)
                    x = kwargs.get("x" if annotate == "y" else "y")
                    # y = kwargs.get("y")
                    # x = kwargs.get("x")
                    x_align = kwargs.get("x_align", "median")
                    order = kwargs.get("order", orig_data[y].unique())
                    kwargs["order"] = order
                    data = orig_data.copy()
                    group = []

                    if not is_categorical_dtype(data[y]):
                        data[y] = pd.Categorical(
                            data[y], categories=order, ordered=True
                        )
                        group.append(y)
                    if (
                        hue is not None
                        and hue_order is not None
                        and not is_categorical_dtype(data[hue])
                    ):
                        data[hue] = pd.Categorical(
                            data[hue], categories=hue_order, ordered=True
                        )
                        group.append(hue)
                    data = data.sort_values(by=group)  # .reset_index(drop=True)
                    if "count" in data.columns:
                        counts = data.groupby(
                            group, as_index=False, sort=True, observed=False
                        ).agg(n=("count", "sum"), x_align=(x, x_align))
                    else:
                        counts = data.groupby(
                            group, as_index=False, sort=True, observed=False
                        ).agg(n=(y, "count"), x_align=(x, x_align))
                    kwargs["data"] = data
                    g: Axes = func(*args, **kwargs)

                    # for patch, group in zip(g.patches, counts.index, strict=False):
                    #     index = counts.loc[group]
                    #     # print(index)
                    #     n = int(index.n)
                    #     # get the center of the box
                    #     print(patch.get_label())
                    #     x = patch.get_x() + patch.get_width() / 2
                    #     y = (
                    #         patch.get_y() + patch.get_height() + 0.05
                    #     )  # just above the box
                    #     # patch doesnt have get_x, get_y, or get_width or get_height
                    #     g.text(
                    #         x,
                    #         y,
                    #         f"PATCH CENTER {x},{y}",
                    #         ha="center",
                    #         va="center",
                    #         fontsize=10,
                    #         color="black",
                    #         bbox={
                    #             "boxstyle": "round,pad=0.2",
                    #             "fc": "white",
                    #             "ec": "none",
                    #             "alpha": 0.8,
                    #         },
                    #     )
                    box_spacing = 1 / len(hue_order) if hue_order else 1

                    previous = None
                    -(box_spacing * 2)
                    for i, group in enumerate(counts.index):
                        index = counts.loc[group]
                        if previous is not None and group != previous:
                            pass
                        # print(index.Model)
                        n = int(index.n)
                        offset = (
                            i * box_spacing
                        )  # - 0.25  # previous_offset + box_spacing + extra_space
                        # print(
                        #     index.Model,
                        #     previous_offset,
                        #     box_spacing,
                        #     extra_space,
                        #     "=",
                        #     offset,
                        #     n,
                        # )
                        if annotate == "y":
                            x_loc = index.x_align + 0.5
                            y_loc = offset
                        elif annotate == "x":
                            x_loc = offset
                            y_loc = index.x_align * 1.025
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
                        previous = group  # index.Model
                    return g, counts
                case _:
                    msg = "Magic annotate not supported"
                    raise ValueError(msg)
        ax = func(*args, **kwargs)

        self.ax = ax if isinstance(ax, Axes) else plt.gca()

        return self.ax

    def _close(self) -> None:
        backend = mpl.get_backend()
        is_ipympl = backend.startswith("module://ipympl")
        if not is_ipympl:
            plt.close()

    def _draw(self) -> None:
        if self.x_label is not None:
            self.ax.set_xlabel(self.x_label, fontsize=self.font_size)

        if self.y_label is not None:
            self.ax.set_ylabel(self.y_label, fontsize=self.font_size)

        if self.title:
            if isinstance(self.title, dict):
                # if "fontsize" not in self.title:
                #     self.title["fontsize"] = (
                #         self.title_size or plt.rcParams["axes.titlesize"]
                #     )
                self.ax.set_title(**self.title)
            else:
                self.ax.set_title(self.title, fontsize=self.title_size)

        if self.ticks:
            self.ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"])
            self.ax.tick_params(axis="y", labelsize=plt.rcParams["ytick.labelsize"])

        if self.despine:
            sns.despine(left=self.despine)

        if self.legend:
            # if self.ax is None:
            #     xlegend = plt.legend(
            #         title=self.legend_title,
            #         frameon=False,
            #         loc="best",
            #         fontsize=plt.rcParams["legend.fontsize"],
            #     )
            #     if self.legend_outside:
            #         xlegend.set_bbox_to_anchor((1.05, 1.0))
            #         xlegend.borderaxespad = 0.0
            #     if self.sketch:
            #         for text in xlegend.get_texts():
            #             text.set_fontfamily("xkcd Script")
            # else:
            xlegend = self.ax.legend(
                **self.legend
                # title=self.legend_title,
                # frameon=True,
                # loc=self.legend_loc,
                # fontsize=plt.rcParams["legend.fontsize"],
            )
            if self.legend_outside and self.legend_top:
                msg = "Cannot have legend_outside and legend_top both True"
                raise ValueError(msg)

            if self.legend_top:
                xlegend.set_bbox_to_anchor((0.5, 1.15))
                xlegend._loc = 9  # 'upper center'
            if self.legend_outside:
                xlegend.set_bbox_to_anchor((1.05, 1.0))
                xlegend.borderaxespad = 0.0
            if self.sketch:
                for text in xlegend.get_texts():
                    text.set_fontfamily("xkcd Script")

        if self.grid:
            self.ax.grid(
                visible=True,
                linestyle="dashed",
                alpha=0.6,
                axis=self.grid,
            )
            # plt.grid(visible=True, linestyle="dashed", alpha=0.6, axis=self.grid)

        # self.figure.tight_layout()
        # self.figure.show()
        if self.tight_layout:
            plt.tight_layout()
        # print("Drawing figure...")
        plt.show()
        self._drawn = True

    def histplot(
        self,
        data: DataFrame,
        x: str,
        hue: str,
        stat: Literal[
            "count", "frequency", "probability", "percent", "density"
        ] = "percent",
        element: Literal["bars", "step", "poly"] = "step",
        common_bins: bool = True,
        common_norm: bool = True,
        **kwargs: dict[str, Any],
    ):
        return sns.histplot(
            data=data,
            x=x,
            hue=hue,
            common_bins=common_bins,
            common_norm=common_norm,
            multiple="layer",
            element=element,
            stat=stat,
            **kwargs,
        )

    def save_fig(self, output_path: Path | str, name: str):
        if isinstance(output_path, str):
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
        self._enter_contexts()
        # print(plt.rcParams)
        if self.exited:
            self.figure.savefig(pdf_filename)  # , dpi=300, bbox_inches="tight")
            self.figure.savefig(
                png_filename, transparent=True
            )  # dpi=300, bbox_inches="tight", transparent=True
            # )
        else:
            warnings.warn(
                "Save_fig invoked before exit, figure incomplete", stacklevel=2
            )
            plt.savefig(pdf_filename)  # , dpi=300, bbox_inches="tight")
            plt.savefig(
                png_filename, transparent=True
            )  # dpi=300, bbox_inches="tight", transparent=True)
        self._exit_contexts()
