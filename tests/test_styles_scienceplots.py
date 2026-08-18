"""Unit and pytest-mpl coverage for the bundled SciencePlots styles.

Regenerate visual baselines after an intentional style change with::

    pytest --mpl-generate-path=tests/baseline \
        tests/test_styles_scienceplots.py::test_scienceplots_style_is_visible

Then verify them with the same test selection and ``--mpl``.
"""

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns

from plotcontext.styles import SCIENCEPLOTS_STYLES, resolve_styles

STYLE_NAMES = tuple(SCIENCEPLOTS_STYLES)


@pytest.fixture
def _standard_color_codes(monkeypatch):
    """Isolate styles from Seaborn's process-global shorthand color remapping."""
    colors = mpl.colors.colorConverter.colors
    for code in "bgrmyck":
        # Register each current value for restoration during fixture teardown.
        monkeypatch.setitem(colors, code, colors[code])
    sns.set_color_codes("reset")


def _style_summary(style_name: str) -> str:
    """Describe settings that cannot be rendered portably without external tools."""
    style = SCIENCEPLOTS_STYLES[style_name]
    details = []

    if cycle := style.get("axes.prop_cycle"):
        details.append(f"cycle={len(cycle)}")
    if family := style.get("font.family"):
        details.append(f"font={','.join(family)}")
    if serif := style.get("font.serif"):
        details.append(f"serif={','.join(serif)}")
    if mathtext := style.get("mathtext.fontset"):
        details.append(f"mathtext={mathtext}")
    if "text.usetex" in style:
        details.append(f"usetex={style['text.usetex']}")
    if preamble := style.get("text.latex.preamble"):
        packages = re.findall(r"\\usepackage(?:\[[^]]+\])?\{([^}]+)\}", preamble)
        details.append(f"tex={','.join(packages)}")
    if style.get("axes.grid"):
        details.append("grid=on")

    return " | ".join(details) or "inherits the science base style"


def test_all_scienceplots_styles_are_ported_and_valid():
    expected_names = {
        "bright",
        "cjk-jp-font",
        "cjk-kr-font",
        "cjk-sc-font",
        "cjk-tc-font",
        "grid",
        "high-contrast",
        "high-vis",
        "ieee",
        "latex-sans",
        "light",
        "muted",
        "nature",
        "no-latex",
        "notebook",
        "pgf",
        "retro",
        "russian-font",
        "sans",
        "scatter",
        "science",
        "std-colors",
        "turkish-font",
        "vibrant",
        *(f"discrete-rainbow-{size}" for size in range(1, 24)),
    }

    assert SCIENCEPLOTS_STYLES.keys() == expected_names
    for style in SCIENCEPLOTS_STYLES.values():
        mpl.RcParams(style)


def test_resolve_styles_maps_bundled_names_and_preserves_matplotlib_names():
    resolved = resolve_styles(["science", "default", "ieee"])

    assert resolved == [
        SCIENCEPLOTS_STYLES["science"],
        "default",
        SCIENCEPLOTS_STYLES["ieee"],
    ]


def test_science_and_ieee_styles_cascade_without_global_registration():
    with mpl.style.context(resolve_styles(["science", "ieee"])):
        assert mpl.rcParams["figure.figsize"] == [3.3, 2.5]
        assert mpl.rcParams["font.size"] == 8.0
        assert mpl.rcParams["legend.frameon"] is False


@pytest.mark.parametrize("style_name", STYLE_NAMES)
@pytest.mark.mpl_image_compare(
    style="default",
    savefig_kwargs={"dpi": 100},
)
def test_scienceplots_style_is_visible(style_name, _standard_color_codes):
    """Render colors, line styles, markers, fonts, legends, and grid settings."""
    # SciencePlots styles are designed to cascade from the science base. Keep
    # the rcParams active until pytest-mpl saves the returned figure.
    mpl.rcParams.update(SCIENCEPLOTS_STYLES["science"])
    mpl.rcParams.update(SCIENCEPLOTS_STYLES[style_name])

    # Tests must not require a system TeX installation. Figure dimensions and
    # DPI are fixed so comparisons stay compact and consistent across styles.
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "figure.figsize": [8.0, 3.6],
            "savefig.dpi": 100,
            "text.usetex": False,
        }
    )

    fig, (bar_ax, line_ax) = plt.subplots(1, 2)
    categories = ["A", "B", "C"]
    hues = ["alpha", "beta", "gamma", "delta"]
    values = [
        [2.0, 3.1, 2.6],
        [2.8, 2.3, 3.4],
        [3.6, 2.9, 2.1],
        [2.4, 3.7, 3.0],
    ]
    sns.barplot(
        data={
            "category": categories * len(hues),
            "hue": [hue for hue in hues for _ in categories],
            "value": [value for row in values for value in row],
        },
        x="category",
        y="value",
        hue="hue",
        errorbar=None,
        saturation=1,
        ax=bar_ax,
    )
    bar_ax.set(title="Grouped bars", xlabel="category", ylabel="value")
    bar_ax.legend(title="hue", ncols=2)

    x = np.linspace(0, 2 * np.pi, 25)
    for index, label in enumerate(hues):
        line_ax.plot(x, np.sin(x + index * 0.55) + index * 0.35, label=label)
    line_ax.set(title="Automatic line cycle", xlabel=r"$x / \pi$", ylabel=r"$f(x)$")
    line_ax.legend(ncols=2)

    cycle = list(mpl.rcParams["axes.prop_cycle"])
    swatch_ax = fig.add_axes((0.1, 0.055, 0.8, 0.035))
    for index, properties in enumerate(cycle):
        swatch_ax.barh(
            0,
            1,
            left=index,
            color=properties.get("color", "none"),
            edgecolor="none",
        )
    swatch_ax.set(xlim=(0, len(cycle)), ylim=(-0.5, 0.5))
    swatch_ax.axis("off")

    fig.suptitle(style_name)
    fig.text(0.5, 0.01, _style_summary(style_name), ha="center", fontsize=6)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.82, wspace=0.3)
    return fig
