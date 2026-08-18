from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from types import ModuleType, TracebackType


class FigureContext:
    def __init__(self, **figure_kwargs: Any) -> None:
        self.figure_kwargs = figure_kwargs
        self.figure: Figure | None = None

    def __enter__(self) -> ModuleType:
        self.figure = plt.figure(**self.figure_kwargs)
        return plt

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            if exc_type is None:
                plt.show()
        finally:
            if self.figure is not None:
                plt.close(self.figure)

        return False
