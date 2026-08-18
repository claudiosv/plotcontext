import json

import pytest

from plotcontext.theming.vscode import (
    JSONWithCommentsDecoder,
    _default_extensions_dir,
    _vscode_user_settings_path,
    get_extension_filepath,
    get_rc_params,
    get_theme_name,
    get_token_color,
    set_theme,
)


@pytest.mark.parametrize(
    ("system", "expected_parts"),
    [
        ("Darwin", ("Library", "Application Support", "Code", "User")),
        ("Windows", ("Code", "User")),
        ("Linux", ("Code", "User")),
    ],
)
def test_vscode_user_settings_path_per_platform(monkeypatch, system, expected_parts):
    monkeypatch.setattr("plotcontext.theming.vscode.platform.system", lambda: system)
    path = _vscode_user_settings_path()
    assert path.name == "settings.json"
    for part in expected_parts:
        assert part in path.parts


def test_default_extensions_dir_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VSCODE_EXTENSIONS_DIR", str(tmp_path))
    assert _default_extensions_dir() == tmp_path


@pytest.mark.parametrize("system", ["Darwin", "Windows", "Linux"])
def test_default_extensions_dir_per_platform(monkeypatch, system):
    monkeypatch.delenv("VSCODE_EXTENSIONS_DIR", raising=False)
    monkeypatch.setattr("plotcontext.theming.vscode.platform.system", lambda: system)
    path = _default_extensions_dir()
    assert path.name == "extensions"


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


def test_json_with_comments_decoder_strips_block_comments():
    raw = """
    {
        /* a block comment
           spanning multiple lines */
        "a": 1,
        "b": "/* not a comment, it's a string */",
        /* another */ "c": 2,
    }
    """
    result = json.loads(raw, cls=JSONWithCommentsDecoder)
    assert result == {"a": 1, "b": "/* not a comment, it's a string */", "c": 2}


def test_json_with_comments_decoder_handles_escaped_quotes():
    raw = r"""{"a": "she said \"hi // there\""}"""
    result = json.loads(raw, cls=JSONWithCommentsDecoder)
    assert result == {"a": 'she said "hi // there"'}


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
        "plotcontext.theming.vscode.VSCODE_USER_SETTINGS_PATH", settings_path
    )

    json_settings, theme_name = get_theme_name()
    assert theme_name == "My Theme"
    assert json_settings["workbench.colorTheme"] == "My Theme"


def test_get_theme_name_defaults_to_dark_modern(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({}))

    monkeypatch.setattr(
        "plotcontext.theming.vscode.VSCODE_USER_SETTINGS_PATH", settings_path
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


def test_get_rc_params_token_color_source_uses_function_token(monkeypatch, tmp_path):
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
                    {"scope": ["function"], "settings": {"foreground": "#ff0000"}},
                    {"scope": ["string"], "settings": {"foreground": "#00ff00"}},
                ],
            }
        )
    )

    monkeypatch.setattr(
        "plotcontext.theming.vscode.get_theme_name", lambda: ({}, "dark_modern")
    )
    monkeypatch.setattr("plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path)

    rc = get_rc_params(text_color_source="token", label_color_source="token")
    assert rc["text.color"] == "#ff0000"
    assert rc["axes.labelcolor"] == "#00ff00"
    # Unaffected rc entries keep using the plain editor foreground.
    assert rc["xtick.color"] == "#eeeeee"


def test_get_rc_params_token_color_source_falls_back_when_token_missing(
    monkeypatch, tmp_path
):
    theme_file = tmp_path / "theme-defaults" / "themes" / "dark_modern.json"
    theme_file.parent.mkdir(parents=True)
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
        "plotcontext.theming.vscode.get_theme_name", lambda: ({}, "dark_modern")
    )
    monkeypatch.setattr("plotcontext.theming.vscode.DEFAULT_EXTENSIONS_DIR", tmp_path)

    rc = get_rc_params(text_color_source="token", label_color_source="token")
    assert rc["text.color"] == "#eeeeee"
    assert rc["axes.labelcolor"] == "#eeeeee"


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
        "plotcontext.theming.vscode.get_rc_params",
        lambda **kwargs: {"axes.facecolor": "red"},
    )
    monkeypatch.setattr(
        "plotcontext.theming.vscode.sns.set_style",
        lambda style, rc: captured.update(style=style, rc=rc),
    )

    set_theme()

    assert captured == {"style": "darkgrid", "rc": {"axes.facecolor": "red"}}


def test_set_theme_forwards_color_source_kwargs(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "plotcontext.theming.vscode.get_rc_params",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(
        "plotcontext.theming.vscode.sns.set_style", lambda style, rc: None
    )

    set_theme(text_color_source="token", label_color_source="token")

    assert captured == {"text_color_source": "token", "label_color_source": "token"}
