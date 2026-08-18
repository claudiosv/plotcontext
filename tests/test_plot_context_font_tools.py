import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from plotcontext.plot_context_font_tools import PlotContextFontTools


@pytest.mark.mpl_image_compare(
    style="default", filename="test_plot_context_font_tools_basic_lifecycle.png"
)
def test_basic_lifecycle_creates_and_closes_figure():
    ctx = PlotContextFontTools(title="T", x_label="X", y_label="Y")
    with ctx as returned:
        assert returned is ctx
        assert ctx.figure is not None
        assert plt.fignum_exists(ctx.figure.number)

    assert ctx._exited is True
    assert not plt.fignum_exists(ctx.figure.number)
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_title_dict_variant():
    ctx = PlotContextFontTools(title={"label": "Dict Title"})
    with ctx:
        pass
    assert ctx._finalized is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_legend_and_grid_and_despine():
    ctx = PlotContextFontTools(legend=True, grid="x", despine=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2], label="line")
    assert ctx._finalized is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_sketch_mode():
    ctx = PlotContextFontTools(sketch=True)
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._exited is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_ieee_plot_context():
    ctx = PlotContextFontTools(plot_context="ieee")
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._exited is True
    return ctx.figure


@pytest.mark.mpl_image_compare(style="default")
def test_acm_plot_context():
    ctx = PlotContextFontTools(plot_context="acm")
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    assert ctx._exited is True
    return ctx.figure


def test_call_injects_managed_axis():
    ctx = PlotContextFontTools()
    with ctx:
        calls = {}

        def fake_plot(*, ax):
            calls["ax"] = ax
            return "result"

        result = ctx(fake_plot)
        assert result == "result"
        assert calls["ax"] is ctx.ax


@pytest.mark.mpl_image_compare(style="default")
def test_save_finalizes_and_writes_file(tmp_path):
    ctx = PlotContextFontTools()
    with ctx:
        ctx.ax.plot([1, 2], [1, 2])
    out = tmp_path / "out.pdf"
    ctx.save(str(out))
    assert out.exists()
    return ctx.figure


def test_save_raises_without_figure():
    ctx = PlotContextFontTools()
    with pytest.raises(RuntimeError, match="No figure to save"):
        ctx.save("out.pdf")


def test_reset_rcparams_resets_defaults(monkeypatch):
    mpl.rcParams["font.size"] = 999.0
    PlotContextFontTools.reset_rcparams()
    assert mpl.rcParams["font.size"] == mpl.rcParamsDefault["font.size"]


def test_font_cache_lists_cached_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "fontlist-v330.json").write_text("{}")
    monkeypatch.setattr(mpl, "get_cachedir", lambda: str(tmp_path))

    PlotContextFontTools.font_cache()

    out = capsys.readouterr().out
    assert "fontlist-v330.json" in out
    assert (tmp_path / "fontlist-v330.json").exists()


def test_refresh_font_cache_deletes_cached_files(tmp_path, monkeypatch, capsys):
    cache_file = tmp_path / "fontlist-v330.json"
    cache_file.write_text("{}")
    monkeypatch.setattr(mpl, "get_cachedir", lambda: str(tmp_path))

    import matplotlib.font_manager as fm

    # Register the current fontManager for restoration, since refresh_font_cache
    # reassigns it directly (bypassing monkeypatch's own tracking).
    monkeypatch.setattr(fm, "fontManager", fm.fontManager)
    monkeypatch.setattr(fm, "_load_fontmanager", lambda **kwargs: None)

    PlotContextFontTools.refresh_font_cache()

    assert not cache_file.exists()
    assert "Font cache refreshed" in capsys.readouterr().out


def test_list_available_serif_fonts(capsys):
    PlotContextFontTools.list_available_serif_fonts()
    assert "Found relevant fonts" in capsys.readouterr().out
