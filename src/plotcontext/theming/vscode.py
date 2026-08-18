import json
import re
from pathlib import Path

import seaborn as sns


class JSONWithCommentsDecoder(json.JSONDecoder):
    """JSON decoder that automatically deals with the comments in vscode json esque settings"""

    def __init__(self, **kw):
        super().__init__(**kw)

    def decode(self, s: str):
        lines = []
        for l in s.split("\n"):
            if l.lstrip(" ").startswith("//"):
                continue
            lines.append(re.sub(r'("(?:[^"\\]|\\.)*")|//.*$', lambda m: m.group(1) or "", l))
        s = "\n".join(lines)
        s = re.sub(r",\s*}", "}", s)  # Remove trailing commas
        s = re.sub(r",\s*]", "]", s)  # Remove trailing commas in arrays
        return super().decode(s)


def get_theme_name():
    """gets the active theme name from user's settings"""
    global_settings_path = Path(
        "~/Library/Application Support/Code/User/settings.json"
    ).expanduser()

    with global_settings_path.open("r") as file:
        json_settings = json.load(file, cls=JSONWithCommentsDecoder)

    theme_name = json_settings.get("workbench.colorTheme", None)
    if not theme_name:
        return json_settings, "dark_modern"
    return json_settings, theme_name


# Bundled (built-in) extensions live inside the app bundle on macOS
DEFAULT_EXTENSIONS_DIR = Path(
    "/Applications/Visual Studio Code.app/Contents/Resources/app/extensions"
)
# User-installed extensions
USER_EXTENSIONS_DIR = Path("~/.vscode/extensions").expanduser()


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
                    package_data = json.load(f, cls=JSONWithCommentsDecoder)

                contributes = package_data.get("contributes", {})
                themes = contributes.get("themes", [])
                name = package_data.get("name", "")
                name = package_data.get("id", name)
                name = name.replace("-", " ")
                category = package_data.get("categories", None)
                if theme_name.lower() in name.lower() and category == ["Themes"]:
                    theme_path = package_data["contributes"]["themes"][0].get("path", "")
                    return folder_path / theme_path.lstrip("/\\")
                for theme in themes:
                    label = theme.get("label", "!!!")
                    if theme_name.lower() in label.lower():  # Case-insensitive comparison
                        theme_path = theme.get("path", "")
                        return folder_path / theme_path.lstrip("/\\")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {package_json_path}: {e}")

    raise KeyError("Theme extension folder was not found")


def get_token_color(settings, token):
    try:
        return settings["semanticTokenColors"][token]
    except KeyError:
        pass
    for t in settings["tokenColors"]:
        if token in t.get("scope", []):
            return t["settings"]["foreground"]


def get_rc_params() -> dict:
    """Compute matplotlib rc params from the active VS Code color theme."""
    json_settings, theme_name = get_theme_name()

    if theme_name != "dark_modern":
        extension_path = get_extension_filepath(theme_name)
    else:
        extension_path = (
            DEFAULT_EXTENSIONS_DIR / "theme-defaults" / "themes" / "dark_modern.json"
        )
    with extension_path.open() as file:
        theme_settings = json.load(file, cls=JSONWithCommentsDecoder)

    if "workbench.colorCustomizations" in json_settings.keys():
        if f"[{theme_name}]" in json_settings["workbench.colorCustomizations"].keys():
            theme_settings["colors"].update(
                json_settings["workbench.colorCustomizations"][f"[{theme_name}]"]
            )

    bg_color = theme_settings["colors"].get("notebook.outputContainerBackgroundColor", None)
    if not bg_color:
        bg_color = theme_settings["colors"].get("editor.background", "#1E1E1E")

    text_color = theme_settings["colors"].get("editor.foreground", "#FFFFFF")
    string_color = get_token_color(theme_settings, "string")
    function_color = get_token_color(theme_settings, "keyword")
    comment_color = get_token_color(theme_settings, "comment")

    return {
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "text.color": text_color,  # function_color,
        "axes.labelcolor": text_color,  # string_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "axes.titlecolor": text_color,
        "grid.color": comment_color,
        "axes.edgecolor": text_color,
        "lines.markeredgecolor": bg_color,
    }


def set_theme() -> None:
    """Apply the active VS Code color theme to matplotlib/seaborn via ``sns.set_style``."""
    sns.set_style("darkgrid", rc=get_rc_params())
