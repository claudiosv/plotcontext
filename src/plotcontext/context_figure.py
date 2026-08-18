from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Self

import matplotlib.pyplot as plt
from matplotlib.axis import Tick
from matplotlib.figure import Figure
from matplotlib.text import Text

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType


class ContextFigure(Figure):
    """A Matplotlib figure that shows and closes itself on context exit."""

    def __enter__(self) -> Self:
        self._activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            # Avoid displaying a partially constructed plot after an error.
            if exc_type is None:
                plt.show()
        finally:
            plt.close(self)

        return False  # Never suppress exceptions.

    def _activate(self) -> None:
        """Make this figure pyplot's current figure."""
        plt.figure(self.number)

    @wraps(plt.title)
    def title(
        self,
        label: str,
        fontdict: dict[str, Any] | None = None,
        loc: Literal["left", "center", "right"] | None = None,
        pad: float | None = None,
        *,
        y: float | None = None,
        **kwargs: Any,
    ) -> Text:
        """Set the title of the current axes."""
        self._activate()
        return plt.title(
            label,
            fontdict=fontdict,
            loc=loc,
            pad=pad,
            y=y,
            **kwargs,
        )

    @wraps(plt.xticks)
    def xticks(
        self,
        ticks: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        *,
        minor: bool = False,
        **kwargs: Any,
    ) -> tuple[list[Tick], list[Text]]:
        """Get or set the current axes' x-axis tick locations and labels."""
        self._activate()
        return plt.xticks(
            ticks=ticks,
            labels=labels,
            minor=minor,
            **kwargs,
        )


def figure(*args: Any, **kwargs: Any) -> ContextFigure:
    """Create a context-managed Matplotlib figure."""
    kwargs["FigureClass"] = ContextFigure
    return plt.figure(*args, **kwargs)  # type: ignore[return-value]
