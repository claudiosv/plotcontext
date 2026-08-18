from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import emoji
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes

EMOJI_TTC = Path("/System/Library/Fonts/Apple Color Emoji.ttc")

_STRIKE_SIZES = (160, 96, 64, 52, 48, 40, 32, 26, 20)


def _font(px: int) -> ImageFont.FreeTypeFont:
    for size in (px, *_STRIKE_SIZES):
        try:
            return ImageFont.truetype(EMOJI_TTC, size=size)
        except OSError:
            continue
    msg = f"no usable strike size in {EMOJI_TTC}"
    raise OSError(msg)


@lru_cache(maxsize=512)
def emoji_image(char: str, px: int = 160) -> npt.NDArray[np.uint8]:
    font = _font(px)
    pad = px // 4
    img = Image.new("RGBA", (2 * (px + pad), px + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad, pad), char, font=font, embedded_color=True)
    box = img.getbbox()
    if box is None:
        msg = f"font rendered no pixels for {char!r}"
        raise ValueError(msg)
    return np.asarray(img.crop(box))


def set_emoji_xticklabels(
    ax: Axes,
    *,
    zoom: float = 0.12,
    offset_points: float = -15.0,
    text_pad: float = 18.0,
    text_kwargs: dict[str, Any] | None = None,
) -> list[AnnotationBbox]:
    """Place color-emoji "tick labels" below the x axis of ``ax``.

    Automatically searches existing x-tick labels for emojis. If found,
    the emoji is extracted, rendered as a high-res image (AnnotationBbox),
    and removed from the text string. Any remaining text is pushed down
    by ``text_pad`` points.
    """
    # Force a canvas draw to ensure Matplotlib has computed dynamic tick labels
    ax.figure.canvas.draw_idle()

    ticks = ax.get_xticks()
    labels = [t.get_text() for t in ax.get_xticklabels()]

    artists = []
    new_labels = []

    for x, text in zip(ticks, labels, strict=True):
        found_emojis = emoji.emoji_list(text)

        if found_emojis:
            # Extract the first emoji found in the tick label
            char = found_emojis[0]["emoji"]
            offset = OffsetImage(
                emoji_image(char), zoom=zoom, interpolation="lanczos", resample=True
            )

            # Strip the emoji out, leaving just the text (if any was present)
            cleaned_text = emoji.replace_emoji(text, replace="").strip()
            new_labels.append(cleaned_text)
        else:
            # No emoji found, keep the original text
            offset = TextArea(text, textprops={"size": zoom * 125})
            new_labels.append("")

        ab = AnnotationBbox(
            offset,
            (x, 0),
            xybox=(0, offset_points),
            xycoords=("data", "axes fraction"),
            boxcoords="offset points",
            frameon=False,
            annotation_clip=False,
        )
        ax.add_artist(ab)
        artists.append(ab)
    # Lock in the tick positions before setting labels to suppress Matplotlib warnings
    ax.set_xticks(ticks)
    ax.set_xticklabels(new_labels, **(text_kwargs or {}))
    ax.tick_params(axis="x", pad=text_pad)

    return artists
