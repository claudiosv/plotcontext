import matplotlib as mpl

from plotcontext.styles import SCIENCEPLOTS_STYLES, resolve_styles


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
