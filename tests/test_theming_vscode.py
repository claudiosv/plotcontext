import json

import pytest

from plotcontext.theming.vscode import (
    JSONWithCommentsDecoder,
    get_extension_filepath,
    get_rc_params,
    get_theme_name,
    get_token_color,
    set_theme,
)


def test_json_with_comments_decoder_strips_line_comments():
    raw = """
    {
        // a leading comment
        "a": 1, // trailing comment
        "b": "// not a comment, it's a string",
        "c": [1, 2, 3,],
    }
    """
    result = json.loads(raw, cls=JSONWithCommentsDecoder)
    assert result == {"a": 1, "b": "// not a comment, it's a string", "c": [1, 2, 3]}


def test_get_token_color_prefers_semantic_token_colors():
    settings = {
        "semanticTokenColors": {"string": "#111111"},
        "tokenColors": [{"scope": ["string"], "settings": {"foreground": "#222222"}}],
    }
    assert get_token_color(settings, "string") == "#111111"


def test_get_token_color_falls_back_to_token_colors():
    settings = {
        "semanticTokenColors": {},
        "tokenColors": [
            {"scope": ["comment"], "settings": {"foreground": "#333333"}},
        ],
    }
    assert get_token_color(settings, "comment") == "#333333"


def test_get_token_color_returns_none_when_not_found():
    settings = {"semanticTokenColors": {}, "tokenColors": []}
    assert get_token_color(settings, "missing") is None


def test_get_theme_name_reads_configured_theme(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"workbench.colorTheme": "My Theme"}))

    monkeypatch.setattr(
        "plotcontext.theming.vscode.Path.expanduser", lambda self: settings_path
    )

    json_settings, theme_name = get_theme_name()
    assert theme_name == "My Theme"
    assert json_settings["workbench.colorTheme"] == "My Theme"


def test_get_theme_name_defaults_to_dark_modern(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({}))

    monkeypatch.setattr(
        "plotcontext.theming.vscode.Path.expanduser", lambda self: settings_path
    )

    _json_settings, theme_name = get_theme_name()
    assert theme_name == "dark_modern"


def test_get_extension_filepath_matches_by_category(tmp_path, monkeypatch):
    ext_dir = tmp_path / "some.theme.ext"
    ext_dir.mkdir()
    (ext_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "cool-theme",
                "categories": ["Themes"],
                "contributes": {"themes": [{"path": "./themes/cool.json"}]},
            }
        )
    )

    monkeypatch.setattr("plotcontext.theming.vscode.USER_EXTENSIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path / "nonexistent"
    )

    path = get_extension_filepath("cool theme")
    assert path == ext_dir / "themes/cool.json"


def test_get_extension_filepath_matches_by_label(tmp_path, monkeypatch):
    ext_dir = tmp_path / "some.other.ext"
    ext_dir.mkdir()
    (ext_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "bundle",
                "contributes": {
                    "themes": [
                        {"label": "Special Theme", "path": "./themes/special.json"}
                    ]
                },
            }
        )
    )

    monkeypatch.setattr("plotcontext.theming.vscode.USER_EXTENSIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path / "nonexistent"
    )

    path = get_extension_filepath("special theme")
    assert path == ext_dir / "themes/special.json"


def test_get_extension_filepath_raises_when_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("plotcontext.theming.vscode.USER_EXTENSIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path / "nonexistent"
    )

    with pytest.raises(KeyError):
        get_extension_filepath("does not exist")


def test_get_extension_filepath_skips_invalid_json(tmp_path, monkeypatch, capsys):
    ext_dir = tmp_path / "broken.ext"
    ext_dir.mkdir()
    (ext_dir / "package.json").write_text("{not valid json,,,")

    monkeypatch.setattr("plotcontext.theming.vscode.USER_EXTENSIONS_DIR", tmp_path)
    monkeypatch.setattr(
        "plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path / "nonexistent"
    )

    with pytest.raises(KeyError):
        get_extension_filepath("anything")

    assert "Error decoding JSON" in capsys.readouterr().out


def test_get_rc_params_builds_style_dict_for_default_theme(monkeypatch, tmp_path):
    theme_file = tmp_path / "theme-defaults" / "themes" / "dark_modern.json"
    theme_file.parent.mkdir(parents=True)
    theme_file.write_text(
        json.dumps(
            {
                "colors": {
                    "editor.background": "#0f0f0f",
                    "editor.foreground": "#eeeeee",
                },
                "tokenColors": [
                    {"scope": ["comment"], "settings": {"foreground": "#555555"}},
                ],
            }
        )
    )

    monkeypatch.setattr(
        "plotcontext.theming.vscode.get_theme_name", lambda: ({}, "dark_modern")
    )
    monkeypatch.setattr("plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path)

    rc = get_rc_params()
    assert rc["axes.facecolor"] == "#0f0f0f"
    assert rc["text.color"] == "#eeeeee"
    assert rc["grid.color"] == "#555555"


def test_get_rc_params_applies_color_customizations(monkeypatch, tmp_path):
    theme_file = tmp_path / "custom_theme.json"
    theme_file.write_text(
        json.dumps(
            {
                "colors": {
                    "editor.background": "#0f0f0f",
                    "editor.foreground": "#eeeeee",
                },
                "tokenColors": [],
            }
        )
    )

    monkeypatch.setattr(
        "plotcontext.theming.vscode.get_theme_name",
        lambda: (
            {
                "workbench.colorCustomizations": {
                    "[Custom]": {"editor.background": "#000000"}
                }
            },
            "Custom",
        ),
    )
    monkeypatch.setattr(
        "plotcontext.theming.vscode.get_extension_filepath", lambda _name: theme_file
    )

    rc = get_rc_params()
    assert rc["axes.facecolor"] == "#000000"


def test_set_theme_calls_sns_set_style(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "plotcontext.theming.vscode.get_rc_params", lambda: {"axes.facecolor": "red"}
    )
    monkeypatch.setattr(
        "plotcontext.theming.vscode.sns.set_style",
        lambda style, rc: captured.update(style=style, rc=rc),
    )

    set_theme()

    assert captured == {"style": "darkgrid", "rc": {"axes.facecolor": "red"}}
