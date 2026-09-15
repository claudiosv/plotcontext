import matplotlib.pyplot as plt
import pytest

from plotcontext.plot_context_builder import ModuleProxy, Plot, SketchConfig


def test_fluent_setters_return_self_and_chain():
    plot = (
        Plot()
        .title("T")
        .labels(x="X", y="Y")
        .size(4, 3, scale=1.5)
        .dpi(150)
        .context("talk")
        .style("darkgrid")
        .extra_styles("seaborn-v0_8")
        .palette("muted", mapping={"a": 0})
        .font_scale(1.2)
        .rc({"font.size": 10})
        .sketch(scale=1.0)
        .legend()
        .grid("y")
        .despine()
        .log(x=True, y=True)
        .tight()
        .debug()
    )
    assert isinstance(plot, Plot)
    assert plot._title == "T"
    assert plot._xlabel == "X"
    assert plot._ylabel == "Y"
    assert plot._fig_x == 4
    assert plot._fig_y == 3
    assert plot._fig_scale == 1.5
    assert plot._dpi == 150
    assert plot._context == "talk"
    assert plot._style == "darkgrid"
    assert "seaborn-v0_8" in plot._extra_styles
    assert plot._palette_name == "muted"
    assert plot._color_map_spec == {"a": 0}
    assert plot._font_scale == 1.2
    assert plot._rc["font.size"] == 10
    assert plot._sketch == {"scale": 1.0}
    assert plot._legend_kwargs == {}
    assert plot._grid_axis == "y"
    assert plot._despine is True
    assert plot._log_x is True
    assert plot._log_y is True
    assert plot._tight is True
    assert plot._debug is True


def test_title_with_kwargs_becomes_dict():
    plot = Plot().title("T", loc="left")
    assert plot._title == {"label": "T", "loc": "left"}


def test_slides_applies_slides_rc():
    plot = Plot().slides()
    assert plot._rc["figure.figsize"] == (10.0, 5.625)
    assert plot._rc["figure.dpi"] == 300


def test_legend_outside_and_top_conflict_raises():
    with pytest.raises(ValueError, match="cannot be both"):
        Plot().legend(outside=True, top=True)


def test_no_figure_skips_figure_creation():
    plot = Plot().no_figure()
    with pytest.warns(UserWarning, match="No ax found"), plot as p:
        assert p.figure is None
        assert p.ax is None


@pytest.mark.mpl_image_compare(
    style="default", filename="test_plot_context_builder_basic_lifecycle.png"
)
def test_basic_lifecycle_creates_and_closes_figure():
    plot = Plot()
    with plot as p:
        assert p.figure is not None
        assert plt.fignum_exists(p.figure.number)
        p.ax.plot([1, 2, 3], [1, 2, 3])

    assert plot.exited is True
    assert not plt.fignum_exists(plot.figure.number)
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_ieee_context_style():
    plot = Plot().context("ieee")
    with plot as p:
        p.ax.plot([1, 2, 3], [1, 2, 3])
    assert plot.exited is True
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_sns_proxy_injects_ax_and_palette():
    plot = Plot().palette("colorblind", mapping={"a": 0, "b": 1})
    with plot as p:
        with pytest.warns(UserWarning, match="Ignoring `palette`"):
            result_ax = p.sns.lineplot(x=[1, 2, 3], y=[1, 4, 9])
        assert result_ax is p.ax
    return plot.figure


def test_plt_proxy_injects_and_tracks_ax():
    plot = Plot()
    with plot as p:
        result_ax = p.plt.gca()
        assert result_ax is p.ax


@pytest.mark.mpl_image_compare(style="default")
def test_plt_proxy_plot_does_not_misinject_ax_kwarg():
    """`plt.plot` (and friends) never accept `ax=`; regression test for a
    bug where the has-**kwargs fallback wrongly forwarded `ax=` into
    Artist property kwargs, raising ``Line2D.set() got an unexpected
    keyword argument 'ax'``.
    """
    plot = Plot()
    with plot as p:
        (line,) = p.plt.plot([1, 2, 3], [1, 4, 9])
        assert line.axes is p.ax
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_draw_applies_labels_title_scales_legend_grid():
    plot = (
        Plot()
        .title("T")
        .labels(x="X", y="Y")
        .log(x=True, y=True)
        .legend()
        .grid("both")
        .despine()
        .tight()
    )
    with plot as p:
        p.ax.plot([1, 2, 3], [1, 2, 3], label="line")

    assert plot._drawn is True
    assert plot.ax.get_xlabel() == "X"
    assert plot.ax.get_ylabel() == "Y"
    assert plot.ax.get_xscale() == "log"
    assert plot.ax.get_yscale() == "log"
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_draw_legend_outside_positions_bbox():
    plot = Plot().legend(outside=True)
    with plot as p:
        p.ax.plot([1, 2], [1, 2], label="line")
    assert plot._drawn is True
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_draw_legend_top_with_sketch_sets_font():
    plot = Plot().legend(top=True).sketch()
    with plot as p:
        p.ax.plot([1, 2], [1, 2], label="line")
    assert plot._drawn is True
    return plot.figure


def test_draw_skips_when_no_ax():
    plot = Plot().no_figure()
    with pytest.warns(UserWarning, match="No ax found"), plot:
        pass
    assert plot._drawn is False


def test_no_ax_warns_on_exit():
    plot = Plot().no_figure()
    with pytest.warns(UserWarning, match="No ax found"):
        with plot:
            pass


@pytest.mark.mpl_image_compare(style="default")
def test_save_writes_pdf(tmp_path):
    plot = Plot()
    with plot as p:
        p.ax.plot([1, 2], [1, 2])
    out = tmp_path / "out.pdf"
    plot.save(out)
    assert out.exists()
    return plot.figure


def test_save_raises_without_figure():
    plot = Plot().no_figure()
    with pytest.raises(RuntimeError, match="No figure to export"):
        plot.save("out.pdf")


@pytest.mark.mpl_image_compare(style="default")
def test_save_fig_writes_timestamped_pdf_and_png(tmp_path):
    plot = Plot()
    with plot as p:
        p.ax.plot([1, 2], [1, 2])
    plot.save_fig(tmp_path, "name/with/slash")
    saved = list(tmp_path.glob("name_with_slash_*.pdf")) + list(
        tmp_path.glob("name_with_slash_*.png")
    )
    assert len(saved) == 2
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_save_fig_rejects_file_as_output_path(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    plot = Plot()
    with plot as p:
        p.ax.plot([1, 2], [1, 2])
    with pytest.raises(ValueError, match="not a directory"):
        plot.save_fig(file_path, "name")
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_save_fig_warns_before_exit(tmp_path):
    plot = Plot()
    plot.figure, plot.ax = plt.subplots()
    plot.ax.plot([1, 2], [1, 2])
    with pytest.warns(UserWarning, match="before exit"):
        plot.save_fig(tmp_path, "name")
    return plot.figure


@pytest.mark.mpl_image_compare(style="default")
def test_clip_copies_png_to_clipboard(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "clipin.copy", lambda data, mime: captured.update(data=data, mime=mime)
    )

    plot = Plot()
    with plot as p:
        p.ax.plot([1, 2], [1, 2])
    plot.clip()

    assert captured["mime"] == "image/png"
    assert isinstance(captured["data"], bytes)
    return plot.figure


def test_module_proxy_passes_through_non_callable_attrs():
    proxy = ModuleProxy(plt, context=None)
    assert proxy.rcParams is plt.rcParams


def test_accepts_kwarg_excludes_no_ax_funcs():
    plot = Plot()

    def relplot(ax=None, **kwargs):
        return kwargs

    relplot.__name__ = "relplot"
    assert plot._accepts_kwarg(relplot, "ax") is False


def test_accepts_kwarg_handles_unintrospectable_callable():
    plot = Plot()
    assert plot._accepts_kwarg(sum, "ax") is False


def test_sketch_config_is_typed_dict():
    config: SketchConfig = {"scale": 1.0, "length": 100.0, "randomness": 2.0}
    assert config["scale"] == 1.0
