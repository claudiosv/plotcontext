import matplotlib.pyplot as plt
import pytest

from plotcontext.context_figure import ContextFigure, figure


def test_figure_factory_returns_context_figure():
    fig = figure()
    assert isinstance(fig, ContextFigure)
    plt.close(fig)


def test_enter_exit_shows_and_closes_on_success():
    fig = figure()
    with fig as ctx:
        assert ctx is fig
        assert plt.fignum_exists(fig.number)

    assert not plt.fignum_exists(fig.number)


def test_exit_closes_without_show_on_exception():
    fig = figure()
    with pytest.raises(ValueError, match="boom"):
        with fig:
            raise ValueError("boom")

    assert not plt.fignum_exists(fig.number)


def test_title_activates_and_sets_title():
    fig = figure()
    with fig as ctx:
        text = ctx.title("Hello")
        assert text.get_text() == "Hello"


def test_xticks_activates_and_returns_ticks_and_labels():
    fig = figure()
    with fig as ctx:
        ax = fig.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        ticks, labels = ctx.xticks([0, 1, 2], ["a", "b", "c"])
        assert [t.get_text() for t in labels] == ["a", "b", "c"]
