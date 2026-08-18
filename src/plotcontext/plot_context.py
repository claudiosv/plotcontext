"""Abstract base class defining the plotting context-manager interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AbstractPlotContext(ABC):
    """
    Base class for context managers that set up a Seaborn-styled Matplotlib plot.

    Subclasses are responsible for creating the figure/axes on ``__enter__``,
    applying any deferred styling in ``_draw``, and releasing the figure in
    ``_close``.
    """

    @staticmethod
    @abstractmethod
    def slides_rc_params() -> dict[Any, Any]:
        """Return rcParams tuned for the standard 16:9 presentation slide layout."""

    @abstractmethod
    def __enter__(self) -> Any:
        """Enter the plotting context, creating the figure/axes."""

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Finalize the plot and tear down the context, preserving exceptions."""

    @abstractmethod
    def save(self, filename: str, **save_kwargs: Any) -> None:
        """Save the current figure to ``filename`` with the given savefig kwargs."""

    @abstractmethod
    def save_fig(self, output_path: Path | str, name: str) -> None:
        """Save the current figure as timestamped PDF/PNG files under output_path."""

    @abstractmethod
    def _draw(self) -> None:
        """Apply any deferred styling (labels, legend, grid, ...) to the axes."""

    @abstractmethod
    def _close(self) -> None:
        """Release the figure owned by this context."""
