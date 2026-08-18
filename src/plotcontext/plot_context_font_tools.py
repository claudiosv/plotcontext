import importlib
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

T = TypeVar("T")


class PlotContextFontTools:
    """Class-based context manager for setting up professional publication plots."""

    def __init__(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        fig_x: float = 3.25,  # Default to ACM/ICML column width
        fig_y: float | None = None,  # Calculated via ratio if None
        height_ratio: float = 0.8,
        fig_scale: float = 1.0,
        font_scale: float = 1.0,
        despine: bool = True,
        sketch: bool = False,
        legend: bool | dict[str, Any] = False,
        grid: Literal["both", "x", "y"] | bool | None = "y",
        sns_style: dict[str, Any]
        | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
        sns_plot_context: Literal["paper", "notebook", "talk", "poster"]
        | None = "paper",
        plot_context: Literal["ieee"] | None = None,
        dpi: int = 300,
    ) -> None:
        # Core dimensions
        self.fig_x = fig_x
        self.fig_y = fig_y or (fig_x * height_ratio)
        self.fig_scale = fig_scale
        self.font_scale = font_scale

        # Labels and Style
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.sns_style = sns_style
        self.sns_plot_context = sns_plot_context
        self.plot_context = plot_context
        self.despine = despine
        self.grid = grid
        self.sketch = sketch
        self.legend = legend
        self.dpi = dpi

        # State
        self.stack = ExitStack()
        self.figure = None
        self.ax = None
        self._exited = False
        self._finalized = False

    def __enter__(self) -> Self:
        # 1. Manage all styling contexts via ExitStack
        # This handles the cleanup of rcParams automatically on __exit__
        try:
            if self.sns_style is not None:
                self.stack.enter_context(sns.axes_style(self.sns_style))
                print(f"Applied Seaborn style: {self.sns_style}")

            # 2. The Scaling Context: Seaborn Plotting Context
            # This replaces the 'context' part of set_theme (e.g., paper, talk)
            # and scales fonts/lines.
            if self.sns_plot_context is not None:
                # We handle IEEE separately to ensure correct font scaling
                self.stack.enter_context(
                    sns.plotting_context(
                        context=self.sns_plot_context, font_scale=self.font_scale
                    )
                )
                print(
                    f"Applied Seaborn plotting context: {self.sns_plot_context} "
                    f"with font scale {self.font_scale}"
                )

            # 3. Specific Overrides: IEEE/Science
            if self.plot_context == "ieee":
                self.stack.enter_context(plt.style.context(["science", "ieee"]))
                print(f"Applied plot context: {self.plot_context}")

            if self.plot_context == "acm":
                # 4. The "Gold Standard": ICML/ACM Style
                # This is entered last to ensure your exact figure size,
                # LaTeX fonts, and PDF settings override everything else.
                # icml_params = get_icml_style()
                # icml_params.update({
                #     "figure.figsize": (self.fig_x, self.fig_y / self.fig_x)
                # })
                icml_params = {
                    "axes.labelsize": 9,
                    "axes.titlepad": 0,
                    "axes.titlesize": 9,
                    "figure.constrained_layout.h_pad": 0.02,
                    "figure.constrained_layout.hspace": 0.01,
                    "figure.constrained_layout.use": True,  # Global toggle
                    "figure.constrained_layout.w_pad": 0.02,
                    "figure.dpi": 300,
                    "figure.figsize": (self.fig_x, self.fig_y / self.fig_x),
                    "font.family": "serif",
                    "font.serif": ["Linux Libertine", "Libertine", "DejaVu Serif"],
                    "font.size": 9,
                    "legend.borderaxespad": 0,
                    "legend.borderpad": 0,
                    "legend.fontsize": 9,
                    "lines.markersize": 3,
                    # "pdf.fonttype": 42,
                    # "ps.fonttype": 42,
                    "savefig.bbox": "tight",
                    "savefig.pad_inches": 0.01,
                    "savefig.transparent": True,
                    # "text.usetex": True,
                    # "text.latex.preamble": (
                    #     r"\usepackage{libertine} \usepackage[libertine]{newtxmath}"
                    # ),
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                }
                self.stack.enter_context(plt.rc_context(rc=icml_params))
                # self.stack.enter_context(
                #     icml_style(width=self.fig_x, height_ratio=self.fig_y / self.fig_x)
                # )
                print(
                    f"Applied ACM/ICML style with figure size {self.fig_x}x{self.fig_y}"
                )
                print(f"ICML style params:\n{icml_params}")

            if self.sketch:
                self.stack.enter_context(plt.xkcd())

            # 2. Initialize the Figure and Axes
            # We use constrained_layout here as it's superior for sub-3.25in plots
            self.figure, self.ax = plt.subplots(
                figsize=(self.fig_x * self.fig_scale, self.fig_y * self.fig_scale),
                # dpi=self.dpi,
                # constrained_layout=True,
            )

        except Exception:
            self.stack.close()  # Clean up any partially entered contexts
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with self.stack:  # This automatically calls __exit__ on all stored contexts
            if exc_type is None and self.ax is not None:
                self._finalize_plot()
                plt.show()

            # If it's not an interactive backend, close to save memory
            if not mpl.get_backend().startswith("module://ipympl"):
                plt.close(self.figure)
        self._exited = True

    def _finalize_plot(self) -> None:
        """Apply labels, titles, and spine formatting."""
        if self.x_label:
            self.ax.set_xlabel(self.x_label)
        if self.y_label:
            self.ax.set_ylabel(self.y_label)

        if self.title:
            if isinstance(self.title, dict):
                self.ax.set_title(**self.title)
            else:
                self.ax.set_title(self.title)

        if self.despine:
            sns.despine(ax=self.ax, trim=True)

        if self.grid:
            self.ax.grid(
                True,
                axis=self.grid if isinstance(self.grid, str) else "both",
                linestyle="--",
                alpha=0.5,
            )

        if self.legend:
            leg_kwargs = self.legend if isinstance(self.legend, dict) else {}
            self.ax.legend(**leg_kwargs)
        self._finalized = True

    def __call__(
        self, func: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> T | tuple[T, pd.DataFrame]:
        """Convenience wrapper to inject the managed axis."""
        if "ax" not in kwargs:
            kwargs["ax"] = self.ax

        return func(*args, **kwargs)

    def save(self, filename: str, **save_kwargs: Any) -> None:
        """Save the current figure to filename with the given savefig kwargs."""
        if self.figure is None:
            raise RuntimeError("No figure to save. Ensure you're within the context.")

        # Default savefig kwargs for publication quality
        default_kwargs = {
            "format": "pdf",
            # "bbox_inches": "tight",
            # "pad_inches": 0.01,
            # "dpi": self.dpi,
        }
        default_kwargs.update(save_kwargs)

        # self.__enter__()
        if not self._finalized:
            with self.stack:  # This automatically calls __exit__ on all stored contexts
                self._finalize_plot()
        self.figure.savefig(filename, **default_kwargs)
        print(f"Saved figure to {filename} with kwargs: {default_kwargs}")

    @staticmethod
    def reset_rcparams() -> None:
        """
        Force resets all rcParams to Matplotlib defaults.
        Useful if a context manager failed to exit properly.
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        # If you use Seaborn, it's also good to reset its specific defaults
        sns.reset_orig()

        try:
            import scienceplots

            importlib.reload(scienceplots)
            # Re-scan the style library so Matplotlib sees the re-registered styles
            plt.style.reload_library()
            print("✔ SciencePlots reloaded and registered.")
        except ImportError:
            print("! SciencePlots not found; skipping reload.")
        print("✔ Matplotlib rcParams have been reset to defaults.")

    @staticmethod
    def font_cache() -> None:
        """
        Clears the font cache and rescans the system for new fonts.
        Run this after installing 'Linux Libertine' or 'DejaVu Serif'.
        """
        # 1. Clear the disk cache
        cache_dir = Path(mpl.get_cachedir()).absolute()
        print(f"🗂 Matplotlib cache directory: {cache_dir}")
        for font_cache in cache_dir.glob("fontlist-v*.json"):
            print(f"🔤 Font cache: {font_cache.name}")

    @staticmethod
    def refresh_font_cache() -> None:
        """
        Clears the font cache and rescans the system for new fonts.
        Run this after installing 'Linux Libertine' or 'DejaVu Serif'.
        """
        # 1. Clear the disk cache
        cache_dir = Path(mpl.get_cachedir()).absolute()
        print(f"🗂 Matplotlib cache directory: {cache_dir}")
        for font_cache in cache_dir.glob("fontlist-v*.json"):
            font_cache.unlink()
            print(f"🗑 Deleted font cache: {font_cache.name}")

        # 2. Rebuild the font manager in memory
        fm._load_fontmanager(try_read_cache=False)

        # 3. Force Matplotlib to see the new manager
        # This re-scans system and user font paths
        fm.fontManager = fm.FontManager()

        print("✨ Font cache refreshed. New system fonts should now be visible.")

    @staticmethod
    def list_available_serif_fonts():
        """Helper to verify if your desired fonts are actually found."""
        serif_fonts = [f.name for f in fm.fontManager.ttflist if f.style == "normal"]
        targets = ["Libertine", "Linux Libertine", "DejaVu Serif"]
        found = [f for f in serif_fonts if any(t in f for t in targets)]
        print(f"🔎 Found relevant fonts: {set(found)}")
