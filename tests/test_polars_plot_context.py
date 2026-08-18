import matplotlib.pyplot as plt
import polars as pl
import pytest

from plotcontext.plot_context import AbstractPlotContext
from plotcontext.polars_plot_context import (
    HistogramStat,
    ModuleProxy,
    PlotContext,
    TickFormatters,
)
from plotcontext.styles.scienceplots import SCIENCEPLOTS_STYLES

# --------------------------------------------------------------------------- #
# Simple value types                                                          #
# --------------------------------------------------------------------------- #


def test_histogram_stat_values():
    assert HistogramStat.COUNT == "count"
    assert HistogramStat.DENSITY == "density"


def test_tick_formatters_expose_matplotlib_classes():
    from matplotlib import ticker

    assert TickFormatters.NullFormatter is ticker.NullFormatter
    assert TickFormatters.PercentFormatter is ticker.PercentFormatter


def test_module_proxy_injects_via_context():
    calls = []

    class FakeContext:
        def _inject_and_call(self, func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return func(*args, **kwargs)

    proxy = ModuleProxy(plt, FakeContext())
    fig = proxy.figure()
    assert calls
    plt.close(fig)


def test_module_proxy_passes_through_non_callables():
    proxy = ModuleProxy(plt, context=None)
    assert proxy.rcParams is plt.rcParams


# --------------------------------------------------------------------------- #
# PlotContext (pandas/polars aware)                                          #
# --------------------------------------------------------------------------- #


def test_plot_context_is_abstract_plot_context():
    assert issubclass(PlotContext, AbstractPlotContext)


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_basic_lifecycle():
    ctx = PlotContext(title="My Title", x_label="X", y_label="Y")
    with ctx as returned:
        assert returned is ctx
        ctx.ax.plot([1, 2, 3], [1, 2, 3])

    assert ctx.exited is True
    assert ctx._drawn is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_title_dict_variant():
    ctx = PlotContext(title={"label": "Dict Title", "loc": "left"})
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.ax.get_title(loc="left") == "Dict Title"
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_despine_and_grid_and_ticks():
    ctx = PlotContext(despine=True, grid="y", ticks=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._drawn is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_log_scales():
    ctx = PlotContext(log_x=True, log_y=True)
    with ctx:
        ctx.ax.plot([1, 2, 3], [1, 2, 3])
    assert ctx.ax.get_xscale() == "log"
    assert ctx.ax.get_yscale() == "log"
    return ctx.figure


def test_plot_context_legend_outside_and_top_conflict():
    ctx = PlotContext(legend=True, legend_outside=True, legend_top=True)
    with pytest.raises(ValueError, match="both True"):
        with ctx:
            ctx.ax.plot([1, 2], [1, 2], label="line")


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_legend_outside():
    ctx = PlotContext(legend=True, legend_outside=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2], label="line")
    assert ctx._drawn is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_legend_top_with_sketch():
    ctx = PlotContext(legend=True, legend_top=True, sketch=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2], label="line")
    assert ctx._drawn is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_tight_layout():
    ctx = PlotContext(tight_layout=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._drawn is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_color_map_injects_palette():
    ctx = PlotContext(color_map={"a": 0, "b": 1})
    with ctx:
        with pytest.warns(UserWarning, match="Ignoring `palette`"):
            result_ax = ctx.sns.scatterplot(x=[1, 2], y=[1, 2])
        assert result_ax is ctx.ax
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_save(tmp_path):
    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    out = tmp_path / "out.pdf"
    ctx.save(str(out))
    assert out.exists()
    return ctx.figure


def test_plot_context_save_raises_without_figure():
    ctx = PlotContext()
    with pytest.raises(RuntimeError, match="No figure to save"):
        ctx.save("out.pdf")


def test_plot_context_save_raises_when_ax_missing_and_not_drawn():
    ctx = PlotContext()
    ctx.figure, ctx.ax = plt.subplots()
    ctx.ax = None
    with pytest.raises(RuntimeError, match="No axes to finalize"):
        ctx.save("out.pdf")
    plt.close(ctx.figure)


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_ieee_style(monkeypatch):
    captured = {}
    real_context = plt.style.context

    def fake_context(styles):
        captured["styles"] = styles
        return real_context([])

    monkeypatch.setattr(plt.style, "context", fake_context)

    ctx = PlotContext(plot_context="ieee")
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.exited is True
    assert captured["styles"] == [
        SCIENCEPLOTS_STYLES["science"],
        SCIENCEPLOTS_STYLES["ieee"],
    ]
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_extra_styles():
    ctx = PlotContext(styles=["seaborn-v0_8"])
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.exited is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_sketch_xkcd_context_entered():
    ctx = PlotContext(sketch={"scale": 2})
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.exited is True
    return ctx.figure


def test_plot_context_enter_failure_closes_stack(monkeypatch):
    ctx = PlotContext()

    def boom(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr("plotcontext.polars_plot_context.plt.subplots", boom)
    with pytest.raises(RuntimeError, match="setup failed"):
        with ctx:
            pass


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_debug_prints_injection_and_call_info(capsys):
    ctx = PlotContext(debug=True)
    with ctx:
        ctx.sns.lineplot(x=[1, 2], y=[1, 2])
    out = capsys.readouterr().out
    assert "accepts" in out
    assert "Called lineplot" in out
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_save_fig_warns_before_exit(tmp_path):
    ctx = PlotContext()
    ctx.figure, ctx.ax = plt.subplots()
    ctx.ax.plot([1, 2], [1, 2])
    with pytest.warns(UserWarning, match="before exit"):
        ctx.save_fig(tmp_path, "name")
    return ctx.figure


def test_plot_context_save_fig_rejects_non_string_name(tmp_path):
    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    with pytest.raises(TypeError, match="Expected name"):
        ctx.save_fig(tmp_path, 123)


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_save_fig_rejects_file_as_output_path(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    with pytest.raises(ValueError, match="not a directory"):
        ctx.save_fig(file_path, "name")
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_clip_copies_png_to_clipboard(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "clipin.copy", lambda data, mime: captured.update(data=data, mime=mime)
    )

    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    ctx.clip()

    assert captured["mime"] == "image/png"
    assert isinstance(captured["data"], bytes)
    assert len(captured["data"]) > 0
    return ctx.figure


def test_plot_context_clip_raises_without_figure(monkeypatch):
    ctx = PlotContext()
    with pytest.raises(RuntimeError, match="No figure to save"):
        ctx.clip()


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_clip_draws_when_not_yet_drawn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "clipin.copy", lambda data, mime: captured.update(data=data, mime=mime)
    )

    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
        assert ctx._drawn is False
        ctx.clip()
        assert ctx._drawn is True

    assert captured["mime"] == "image/png"
    return ctx.figure


def test_plot_context_accepts_kwarg_detects_var_keyword():
    ctx = PlotContext()

    def func(**kwargs):
        return kwargs

    assert ctx._accepts_kwarg(func, "ax") is True


def test_plot_context_accepts_kwarg_excludes_no_ax_funcs():
    ctx = PlotContext()

    def relplot(ax=None, **kwargs):
        return kwargs

    relplot.__name__ = "relplot"
    assert ctx._accepts_kwarg(relplot, "ax") is False


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_annotate_boxplot():
    df = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b", "b"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ctx = PlotContext()
    with ctx:
        ax, counts = ctx.sns.boxplot(data=df, x="value", y="group", annotate="y")
        assert ax is ctx.ax
        assert set(counts["n"].to_list()) == {2, 3}
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_annotate_boxplot_with_hue():
    df = pl.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "sub": ["x"] * 5 + ["y"] * 5 + ["x"] * 5 + ["y"] * 5,
            "value": [
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                2.3,
                2.6,
                2.9,
                3.2,
                3.0,
                3.4,
                3.8,
                4.2,
                4.6,
                4.0,
                4.5,
                5.0,
                5.5,
                6.0,
            ],
        }
    )
    ctx = PlotContext()
    with ctx:
        ax, counts = ctx.sns.boxplot(
            data=df, x="value", y="group", hue="sub", annotate="y"
        )
        assert ax is ctx.ax
        assert set(counts["n"].to_list()) == {5}
        assert [text.get_text() for text in ax.texts] == ["n=5"] * 4
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_annotate_boxplot_with_existing_count_column():
    df = pl.DataFrame(
        {
            "group": ["a", "a", "b"],
            "value": [1.0, 2.0, 3.0],
            "count": [10, 20, 30],
        }
    )
    ctx = PlotContext()
    with ctx:
        ax, counts = ctx.sns.boxplot(data=df, x="value", y="group", annotate="y")
        assert ax is ctx.ax
        assert set(counts["n"].to_list()) == {30}
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_plot_context_annotate_violinplot_with_hue():
    df = pl.DataFrame(
        {
            "group": ["a"] * 12 + ["b"] * 12,
            "sub": ["x"] * 6 + ["y"] * 6 + ["x"] * 6 + ["y"] * 6,
            "value": [
                1.0,
                1.2,
                1.3,
                1.5,
                1.7,
                1.9,
                2.0,
                2.3,
                2.6,
                2.9,
                3.2,
                3.5,
                4.0,
                4.4,
                4.8,
                5.2,
                5.6,
                6.0,
                6.0,
                6.6,
                7.2,
                7.8,
                8.4,
                9.0,
            ],
        }
    )
    ctx = PlotContext()
    with ctx:
        ax, counts = ctx.sns.violinplot(
            data=df,
            x="value",
            y="group",
            hue="sub",
            split=True,
            inner=None,
            cut=0,
            annotate="y",
        )
        assert ax is ctx.ax
        assert counts["n"].to_list() == [6, 6, 6, 6]
        assert [text.get_text() for text in ax.texts] == ["n=6"] * 4
    return ctx.figure


def test_plot_context_annotate_requires_valid_axis():
    df = pl.DataFrame({"group": ["a"], "value": [1.0]})
    ctx = PlotContext()
    with ctx, pytest.raises(ValueError, match="axis"):
        ctx.sns.boxplot(data=df, x="value", y="group", annotate="z")


def test_plot_context_annotate_unsupported_func_raises():
    ctx = PlotContext()
    with ctx, pytest.raises(ValueError, match="not supported"):
        ctx.sns.lineplot(x=[1], y=[1], annotate="y")
