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


def test_plot_context_basic_lifecycle():
    ctx = PlotContext(title="My Title", x_label="X", y_label="Y")
    with ctx as returned:
        assert returned is ctx
        ctx.ax.plot([1, 2, 3], [1, 2, 3])

    assert ctx.exited is True
    assert ctx._drawn is True


def test_plot_context_title_dict_variant():
    ctx = PlotContext(title={"label": "Dict Title", "loc": "left"})
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.ax.get_title(loc="left") == "Dict Title"


def test_plot_context_despine_and_grid_and_ticks():
    ctx = PlotContext(despine=True, grid="y", ticks=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._drawn is True


def test_plot_context_log_scales():
    ctx = PlotContext(log_x=True, log_y=True)
    with ctx:
        ctx.ax.plot([1, 2, 3], [1, 2, 3])
    assert ctx.ax.get_xscale() == "log"
    assert ctx.ax.get_yscale() == "log"


def test_plot_context_legend_outside_and_top_conflict():
    ctx = PlotContext(legend=True, legend_outside=True, legend_top=True)
    with pytest.raises(ValueError, match="both True"):
        with ctx:
            ctx.ax.plot([1, 2], [1, 2], label="line")


def test_plot_context_legend_outside():
    ctx = PlotContext(legend=True, legend_outside=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2], label="line")
    assert ctx._drawn is True


def test_plot_context_legend_top_with_sketch():
    ctx = PlotContext(legend=True, legend_top=True, sketch=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2], label="line")
    assert ctx._drawn is True


def test_plot_context_tight_layout():
    ctx = PlotContext(tight_layout=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._drawn is True


def test_plot_context_color_map_injects_palette():
    ctx = PlotContext(color_map={"a": 0, "b": 1})
    with ctx:
        result_ax = ctx.sns.scatterplot(x=[1, 2], y=[1, 2])
        assert result_ax is ctx.ax


def test_plot_context_save(tmp_path):
    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    out = tmp_path / "out.pdf"
    ctx.save(str(out))
    assert out.exists()


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


def test_plot_context_ieee_style(monkeypatch):
    # "science"/"ieee" styles are registered globally by scienceplots at import
    # time; other tests that reload the style library can drop them, so stub
    # plt.style.context instead of depending on that global mutable state.
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
    assert captured["styles"] == ["science", "ieee"]


def test_plot_context_extra_styles():
    ctx = PlotContext(styles=["seaborn-v0_8"])
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.exited is True


def test_plot_context_sketch_xkcd_context_entered():
    ctx = PlotContext(sketch={"scale": 2})
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx.exited is True


def test_plot_context_enter_failure_closes_stack(monkeypatch):
    ctx = PlotContext()

    def boom(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr("plotcontext.polars_plot_context.plt.subplots", boom)
    with pytest.raises(RuntimeError, match="setup failed"):
        with ctx:
            pass


def test_plot_context_debug_prints_injection_and_call_info(capsys):
    ctx = PlotContext(debug=True)
    with ctx:
        ctx.sns.lineplot(x=[1, 2], y=[1, 2])
    out = capsys.readouterr().out
    assert "accepts" in out
    assert "Called lineplot" in out


def test_plot_context_save_fig_warns_before_exit(tmp_path):
    ctx = PlotContext()
    ctx.figure, ctx.ax = plt.subplots()
    ctx.ax.plot([1, 2], [1, 2])
    with pytest.warns(UserWarning, match="before exit"):
        ctx.save_fig(tmp_path, "name")
    plt.close(ctx.figure)


def test_plot_context_save_fig_rejects_non_string_name(tmp_path):
    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    with pytest.raises(TypeError, match="Expected name"):
        ctx.save_fig(tmp_path, 123)


def test_plot_context_save_fig_rejects_file_as_output_path(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    ctx = PlotContext()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    with pytest.raises(ValueError, match="not a directory"):
        ctx.save_fig(file_path, "name")


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


def test_plot_context_clip_raises_without_figure(monkeypatch):
    ctx = PlotContext()
    with pytest.raises(RuntimeError, match="No figure to save"):
        ctx.clip()


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


def test_plot_context_annotate_boxplot_with_hue():
    df = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "sub": ["x", "y", "x", "y"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ctx = PlotContext()
    with ctx:
        ax, counts = ctx.sns.boxplot(
            data=df, x="value", y="group", hue="sub", annotate="y"
        )
        assert ax is ctx.ax
        assert set(counts["n"].to_list()) == {1}


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


def test_plot_context_annotate_requires_valid_axis():
    df = pl.DataFrame({"group": ["a"], "value": [1.0]})
    ctx = PlotContext()
    with ctx, pytest.raises(ValueError, match="axis"):
        ctx.sns.boxplot(data=df, x="value", y="group", annotate="z")


def test_plot_context_annotate_unsupported_func_raises():
    ctx = PlotContext()
    with ctx, pytest.raises(ValueError, match="not supported"):
        ctx.sns.lineplot(x=[1], y=[1], annotate="y")
