"""Derive matplotlib/seaborn rc params from the active VS Code color theme."""

import os
import platform
from pathlib import Path
from typing import Literal

import jsonc
import seaborn as sns

ColorSource = Literal["text", "token"]

DEFAULT_THEME_NAME = "dark_modern"


def _vscode_user_settings_path() -> Path:
    """Location of VS Code's global ``settings.json`` for the current OS."""
    system = platform.system()
    if system == "Darwin":
        base = Path("~/Library/Application Support").expanduser()
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", "~/AppData/Roaming")).expanduser()
    else:  # Linux and other POSIX systems
        base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return base / "Code" / "User" / "settings.json"


def _default_extensions_dir() -> Path:
    """Best-effort location of VS Code's bundled (built-in) extensions.

    Set the ``VSCODE_EXTENSIONS_DIR`` environment variable to override this,
    e.g. on Linux where the install path varies by package manager.
    """
    if override := os.environ.get("VSCODE_EXTENSIONS_DIR"):
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path(
            "/Applications/Visual Studio Code.app/Contents/Resources/app/extensions"
        )
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", "~/AppData/Local")).expanduser()
        return (
            base / "Programs" / "Microsoft VS Code" / "resources" / "app" / "extensions"
        )
    # Linux: covers the common apt/dnf package path; snap/flatpak installs
    # differ, hence the VSCODE_EXTENSIONS_DIR override above.
    return Path("/usr/share/code/resources/app/extensions")


# Bundled (built-in) extensions, e.g. the "theme-defaults" package.
DEFAULT_EXTENSIONS_DIR = _default_extensions_dir()
# User-installed extensions; same relative layout on macOS, Linux, and Windows.
USER_EXTENSIONS_DIR = Path("~/.vscode/extensions").expanduser()
VSCODE_USER_SETTINGS_PATH = _vscode_user_settings_path()


def get_token_color(settings, token):
    try:
        return settings["semanticTokenColors"][token]
    except KeyError:
        pass
    for t in settings["tokenColors"]:
        if token in t.get("scope", []):
            return t["settings"]["foreground"]


def get_theme_name():
    """Gets the active theme name from the user's VS Code settings."""
    with VSCODE_USER_SETTINGS_PATH.open("r") as file:
        json_settings = jsonc.load(file)

    theme_name = json_settings.get("workbench.colorTheme", None)
    return json_settings, (theme_name or DEFAULT_THEME_NAME)


def get_extension_filepath(theme_name):
    for location in [USER_EXTENSIONS_DIR, DEFAULT_EXTENSIONS_DIR]:
        if not location.is_dir():
            continue
        for folder_path in location.iterdir():
            if not folder_path.is_dir():
                continue

            package_json_path = folder_path / "package.json"
            if not package_json_path.exists():
                continue

            try:
                with package_json_path.open("r", encoding="utf-8") as f:
                    package_data = jsonc.load(f)

                contributes = package_data.get("contributes", {})
                themes = contributes.get("themes", [])
                name = package_data.get("name", "")
                name = package_data.get("id", name)
                name = name.replace("-", " ")
                category = package_data.get("categories", None)
                if theme_name.lower() in name.lower() and category == ["Themes"]:
                    theme_path = package_data["contributes"]["themes"][0].get(
                        "path", ""
                    )
                    return folder_path / theme_path.lstrip("/\\")
                for theme in themes:
                    label = theme.get("label", "!!!")
                    if (
                        theme_name.lower() in label.lower()
                    ):  # Case-insensitive comparison
                        theme_path = theme.get("path", "")
                        return folder_path / theme_path.lstrip("/\\")
            except jsonc.JSONDecodeError as e:
                print(f"Error decoding JSON in {package_json_path}: {e}")

    raise KeyError("Theme extension folder was not found")


def get_rc_params(
    *,
    text_color_source: ColorSource = "text",
    label_color_source: ColorSource = "text",
) -> dict:
    """Compute matplotlib rc params from the active VS Code color theme.

    Parameters
    ----------
    text_color_source : {"text", "token"}
        Color used for ``text.color``. ``"text"`` uses the theme's editor
        foreground; ``"token"`` uses the syntax-highlighting color for
        function tokens, falling back to the editor foreground if the theme
        doesn't define one.
    label_color_source : {"text", "token"}
        Color used for ``axes.labelcolor``, with the same options as
        ``text_color_source`` but sourced from the theme's string tokens.
    """
    json_settings, theme_name = get_theme_name()

    if theme_name != DEFAULT_THEME_NAME:
        extension_path = get_extension_filepath(theme_name)
    else:
        extension_path = (
            DEFAULT_EXTENSIONS_DIR / "theme-defaults" / "themes" / "dark_modern.json"
        )
    with extension_path.open() as file:
        theme_settings = jsonc.load(file)

    if "workbench.colorCustomizations" in json_settings.keys():
        if f"[{theme_name}]" in json_settings["workbench.colorCustomizations"].keys():
            theme_settings["colors"].update(
                json_settings["workbench.colorCustomizations"][f"[{theme_name}]"]
            )

    bg_color = theme_settings["colors"].get(
        "notebook.outputContainerBackgroundColor", None
    )
    if not bg_color:
        bg_color = theme_settings["colors"].get("editor.background", "#1E1E1E")

    text_color = theme_settings["colors"].get("editor.foreground", "#FFFFFF")
    comment_color = get_token_color(theme_settings, "comment")

    resolved_text_color = text_color
    if text_color_source == "token":
        resolved_text_color = get_token_color(theme_settings, "function") or text_color

    resolved_label_color = text_color
    if label_color_source == "token":
        resolved_label_color = get_token_color(theme_settings, "string") or text_color

    return {
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "text.color": resolved_text_color,
        "axes.labelcolor": resolved_label_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "axes.titlecolor": text_color,
        "grid.color": comment_color,
        "axes.edgecolor": text_color,
        "lines.markeredgecolor": bg_color,
    }


def set_theme(
    *,
    text_color_source: ColorSource = "text",
    label_color_source: ColorSource = "text",
) -> None:
    """Apply the active VS Code color theme to matplotlib/seaborn via sns.set_style.

    See `get_rc_params` for `text_color_source` and `label_color_source`.
    """
    rc = get_rc_params(
        text_color_source=text_color_source, label_color_source=label_color_source
    )
    sns.set_style("darkgrid", rc=rc)
