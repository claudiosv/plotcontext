import matplotlib.pyplot as plt
import pytest

from plotcontext.figure_context import FigureContext


def test_enter_creates_and_returns_pyplot():
    ctx = FigureContext()
    with ctx as pyplot_module:
        assert pyplot_module is plt
        assert ctx.figure is not None
        assert plt.fignum_exists(ctx.figure.number)

    assert not plt.fignum_exists(ctx.figure.number)


@pytest.mark.mpl_image_compare(style="default")
def test_figure_kwargs_forwarded():
    ctx = FigureContext(figsize=(3, 2))
    with ctx:
        assert tuple(ctx.figure.get_size_inches()) == (3.0, 2.0)
        ctx.figure.subplots().plot([0, 1, 2], [0, 1, 4])
    return ctx.figure


def test_exception_closes_without_show():
    ctx = FigureContext()
    with pytest.raises(ValueError, match="boom"):
        with ctx:
            raise ValueError("boom")

    assert not plt.fignum_exists(ctx.figure.number)
