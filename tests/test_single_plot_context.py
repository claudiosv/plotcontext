import matplotlib.pyplot as plt
import pytest

from plotcontext.plot_context import AbstractPlotContext
from plotcontext.single_plot_context import SinglePlotContext


def test_single_plot_context_is_abstract_plot_context():
    assert issubclass(SinglePlotContext, AbstractPlotContext)


def test_single_plot_context_creates_and_closes_figure():
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with ctx as (returned_ctx, fig, ax):
        assert returned_ctx is ctx
        assert fig is not None
        assert ax is not None
        ax.plot([1, 2, 3], [1, 2, 3])

    assert ctx.exited is True
    assert not plt.fignum_exists(fig.number)


def test_single_plot_context_set_title():
    with SinglePlotContext(figsize=(4.0, 3.0)) as (ctx, _fig, _ax):
        text = ctx.set_title("hello")
        assert text.get_text() == "hello"


def test_single_plot_context_sns_proxy_injects_ax():
    with SinglePlotContext(figsize=(4.0, 3.0)) as (ctx, _fig, ax):
        result_ax = ctx.sns.lineplot(x=[1, 2, 3], y=[1, 4, 9])
        assert result_ax is ax


def test_single_plot_context_save(tmp_path):
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with ctx as (_ctx, _fig, ax):
        ax.plot([1, 2], [1, 2])
        out = tmp_path / "out.pdf"
        ctx.save(str(out))
    assert out.exists()


def test_single_plot_context_save_fig(tmp_path):
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with ctx as (_ctx, _fig, ax):
        ax.plot([1, 2], [1, 2])
    ctx.save_fig(tmp_path, "myplot")
    saved = list(tmp_path.glob("myplot_*.pdf")) + list(tmp_path.glob("myplot_*.png"))
    assert len(saved) == 2


def test_single_plot_context_activate_raises_without_figure():
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with pytest.raises(RuntimeError, match="no active figure"):
        ctx._activate()


def test_single_plot_context_sketch_mode_updates_rc_params():
    ctx = SinglePlotContext(figsize=(4.0, 3.0), sketch=True)
    assert ctx.rc_params["font.family"] == "sans-serif"


def test_single_plot_context_no_ax_warns_on_exit():
    ctx = SinglePlotContext(figsize=(4.0, 3.0), create_fig=False)
    with pytest.warns(UserWarning, match="No ax found"):
        with ctx:
            pass


def test_single_plot_context_exception_does_not_draw():
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with pytest.raises(ValueError, match="boom"):
        with ctx as (_ctx, _fig, _ax):
            raise ValueError("boom")
    assert ctx._drawn is False


def test_single_plot_context_ieee_style(monkeypatch):
    # "science"/"ieee" styles are registered globally by scienceplots at import
    # time; other tests that reload the style library can drop them, so stub
    # plt.style.context instead of depending on that global mutable state.
    captured = {}
    real_context = plt.style.context

    def fake_context(styles):
        captured["styles"] = styles
        return real_context([])

    monkeypatch.setattr(plt.style, "context", fake_context)

    ctx = SinglePlotContext(figsize=(4.0, 3.0), plot_context="ieee")
    with ctx:
        pass
    assert ctx.exited is True
    assert captured["styles"] == ["science", "ieee"]


def test_single_plot_context_extra_styles():
    ctx = SinglePlotContext(figsize=(4.0, 3.0), styles=["seaborn-v0_8"])
    with ctx:
        pass
    assert ctx.exited is True


def test_single_plot_context_sketch_xkcd_context_entered():
    ctx = SinglePlotContext(figsize=(4.0, 3.0), sketch={"scale": 2})
    with ctx as (_ctx, _fig, ax):
        ax.plot([1, 2], [1, 2])
    assert ctx.exited is True


def test_single_plot_context_enter_failure_closes_stack(monkeypatch):
    ctx = SinglePlotContext(figsize=(4.0, 3.0))

    def boom(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr("plotcontext.single_plot_context.plt.subplots", boom)
    with pytest.raises(RuntimeError, match="setup failed"):
        with ctx:
            pass


def test_single_plot_context_debug_prints_injection_info(capsys):
    ctx = SinglePlotContext(figsize=(4.0, 3.0), debug=True)
    with ctx as (_ctx, _fig, ax):
        ctx.sns.lineplot(x=[1, 2], y=[1, 2])
    out = capsys.readouterr().out
    assert "accepts" in out
    assert "Called lineplot" in out


def test_single_plot_context_save_raises_when_ax_missing_and_not_drawn():
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    ctx.figure, ctx.ax = plt.subplots()
    ctx.ax = None
    with pytest.raises(RuntimeError, match="No axes to finalize"):
        ctx.save("out.pdf")
    plt.close(ctx.figure)


def test_single_plot_context_slides_rc_params():
    rc = SinglePlotContext.slides_rc_params()
    assert rc["figure.figsize"] == (10.0, 5.625)
    assert rc["figure.dpi"] == 300


def test_single_plot_context_color_map_injects_palette():
    ctx = SinglePlotContext(figsize=(4.0, 3.0), color_map={"a": 0, "b": 1})
    with ctx as (_ctx, _fig, _ax):
        result_ax = ctx.sns.scatterplot(x=[1, 2], y=[1, 2])
        assert result_ax is ctx.ax


def test_single_plot_context_tight_layout_and_legend_sketch():
    ctx = SinglePlotContext(figsize=(4.0, 3.0), tight_layout=True, sketch=True)
    with ctx as (_ctx, _fig, ax):
        ax.plot([1, 2], [1, 2], label="line")
        ax.legend()
    assert ctx._drawn is True


def test_single_plot_context_save_fig_rejects_non_string_name(tmp_path):
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with ctx:
        pass
    with pytest.raises(TypeError, match="Expected name"):
        ctx.save_fig(tmp_path, 123)


def test_single_plot_context_save_fig_rejects_file_as_output_path(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    with ctx:
        pass
    with pytest.raises(ValueError, match="not a directory"):
        ctx.save_fig(file_path, "name")


def test_single_plot_context_save_fig_warns_before_exit(tmp_path):
    ctx = SinglePlotContext(figsize=(4.0, 3.0))
    ctx.figure, ctx.ax = plt.subplots()
    with pytest.warns(UserWarning, match="before exit"):
        ctx.save_fig(tmp_path, "name")
    plt.close(ctx.figure)
