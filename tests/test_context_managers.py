import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotcontext.context_managers import auto_show, facet_context, plot_context


def test_plot_context_yields_fig_and_ax_and_closes():
    with plot_context() as (fig, ax):
        assert fig is not None
        assert ax is not None
        assert plt.fignum_exists(fig.number)

    assert not plt.fignum_exists(fig.number)


def test_auto_show_closes_all_figures():
    plt.figure()
    plt.figure()
    with auto_show():
        assert len(plt.get_fignums()) == 2

    assert plt.get_fignums() == []


def test_facet_context_single_plot_grid():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    g = sns.relplot(data=df, x="x", y="y")
    with facet_context(g) as (grid, fig, axes):
        assert grid is g
        assert fig is g.fig
        assert axes is g.ax

    assert not plt.fignum_exists(fig.number)
