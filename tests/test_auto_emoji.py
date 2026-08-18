import numpy as np
import pytest
from PIL import Image, ImageDraw

import plotcontext.auto_emoji as auto_emoji
from plotcontext.auto_emoji import emoji_image, set_emoji_xticklabels


def test_font_falls_back_through_strike_sizes(monkeypatch):
    calls = []

    def fake_truetype(path, size):
        calls.append(size)
        if size != auto_emoji._STRIKE_SIZES[-1]:
            raise OSError("no such strike")
        return "sentinel-font"

    monkeypatch.setattr(auto_emoji.ImageFont, "truetype", fake_truetype)

    result = auto_emoji._font(9999)
    assert result == "sentinel-font"
    assert calls[0] == 9999
    assert calls[-1] == auto_emoji._STRIKE_SIZES[-1]


def test_font_raises_when_no_size_works(monkeypatch):
    def always_fail(path, size):
        raise OSError("nope")

    monkeypatch.setattr(auto_emoji.ImageFont, "truetype", always_fail)

    with pytest.raises(OSError, match="no usable strike size"):
        auto_emoji._font(10)


def test_emoji_image_renders_and_crops(monkeypatch):
    emoji_image.cache_clear()
    real_image_new = Image.new
    real_draw = ImageDraw.Draw

    def fake_font(px):
        return "font"

    def fake_new(mode, size, color):
        return real_image_new(mode, size, color)

    class FakeDraw:
        def __init__(self, img):
            self._img = img

        def text(self, xy, char, font, embedded_color):
            draw = real_draw(self._img)
            draw.rectangle([xy, (xy[0] + 10, xy[1] + 10)], fill=(255, 0, 0, 255))

    monkeypatch.setattr(auto_emoji, "_font", fake_font)
    monkeypatch.setattr(auto_emoji.Image, "new", fake_new)
    monkeypatch.setattr(auto_emoji.ImageDraw, "Draw", FakeDraw)

    arr = emoji_image("😀", px=16)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] > 0
    assert arr.shape[1] > 0
    emoji_image.cache_clear()


def test_emoji_image_raises_when_nothing_rendered(monkeypatch):
    emoji_image.cache_clear()
    real_image_new = Image.new

    monkeypatch.setattr(auto_emoji, "_font", lambda px: "font")
    monkeypatch.setattr(
        auto_emoji.Image,
        "new",
        lambda mode, size, color: real_image_new(mode, size, color),
    )

    class BlankDraw:
        def __init__(self, img):
            pass

        def text(self, *args, **kwargs):
            pass

    monkeypatch.setattr(auto_emoji.ImageDraw, "Draw", BlankDraw)

    with pytest.raises(ValueError, match="no pixels"):
        emoji_image("😀", px=16)
    emoji_image.cache_clear()


@pytest.mark.mpl_image_compare(style="default")
def test_set_emoji_xticklabels_without_emoji(monkeypatch):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["a", "b", "c"])

    artists = set_emoji_xticklabels(ax)

    assert len(artists) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["", "", ""]
    return fig


@pytest.mark.mpl_image_compare(style="default")
def test_set_emoji_xticklabels_with_emoji(monkeypatch):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["🔥 hot", "cold"])

    fake_array = np.zeros((4, 4, 4), dtype=np.uint8)
    monkeypatch.setattr(auto_emoji, "emoji_image", lambda char, px=160: fake_array)

    with pytest.warns(UserWarning, match="missing from font"):
        artists = set_emoji_xticklabels(ax)

    assert len(artists) == 2
    new_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert new_labels[0] == "hot"
    assert new_labels[1] == ""
    return fig
