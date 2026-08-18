"""Backend-awareness helper shared by the plot context managers."""

import matplotlib as mpl


def can_show() -> bool:
    """Whether the active matplotlib backend can actually display a figure.

    ``plt.show()`` on a non-interactive backend (e.g. the headless ``Agg``
    backend used under pytest) does nothing but emit a ``UserWarning``, so
    callers should skip it entirely rather than call it unconditionally.
    """
    return mpl.get_backend().lower() != "agg"
