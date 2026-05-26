"""Image rendering for visual anchor overlays (Set-of-Marks style).

Used by experiments that prime models with anchor bboxes drawn on the
screenshot. The drawing protocol — single magenta color, line width,
label format and position — is fixed and identical across anchors so
no model can be biased by visual hierarchy among anchors.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .coords import BBoxPx


# Visual style. Single highly-saturated color rare in mobile UIs to avoid
# blending with app content. Identical for every anchor.
ANCHOR_COLOR = "#FF00FF"          # magenta
ANCHOR_LINE_WIDTH = 6
LABEL_BG_COLOR = "#FF00FF"
LABEL_TEXT_COLOR = "#FFFFFF"
LABEL_PADDING = 6
DEFAULT_LABEL_FONT_SIZE = 32       # readable at 1080-2400 px screenshots


def _load_font(size: int) -> ImageFont.ImageFont:
    """Try a few common system fonts; fall back to PIL's bitmap default.

    The default PIL font is small and unscalable, so we attempt TrueType
    fonts available on macOS / Linux / Docker images first.
    """
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_anchors(
    image_path: str | Path,
    anchors_px: list[BBoxPx],
    labels: list[str] | None = None,
    color: str = ANCHOR_COLOR,
    line_width: int = ANCHOR_LINE_WIDTH,
    font_size: int = DEFAULT_LABEL_FONT_SIZE,
) -> bytes:
    """Render the screenshot at `image_path` with anchor bboxes drawn over it.

    Each anchor gets a rectangle outline + a labeled tag placed adjacent to
    the bbox (above when there is room, below otherwise) so it never occludes
    the anchor itself. Returns JPEG bytes (quality 95) ready to encode for
    the API.

    `labels` defaults to ['Anchor 1', 'Anchor 2', ...]. Pass a custom list
    if you want different labels.
    """
    if labels is None:
        labels = [f"Anchor {i + 1}" for i in range(len(anchors_px))]
    if len(labels) != len(anchors_px):
        raise ValueError(
            f"labels ({len(labels)}) must match anchors_px ({len(anchors_px)})"
        )

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    for bbox, label in zip(anchors_px, labels):
        x1, y1, x2, y2 = bbox
        # Anchor outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label position: above the bbox if there is room, else below
        # Measure label size
        try:
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = r - l, b - t
        except AttributeError:
            # Older Pillow without textbbox — approximate
            text_w = font_size * len(label) * 0.55
            text_h = font_size

        tag_w = int(text_w + 2 * LABEL_PADDING)
        tag_h = int(text_h + 2 * LABEL_PADDING)

        if y1 - tag_h - 4 >= 0:
            tag_x1, tag_y1 = x1, y1 - tag_h - 4
        else:
            tag_x1, tag_y1 = x1, y2 + 4
        tag_x2, tag_y2 = tag_x1 + tag_w, tag_y1 + tag_h

        # Solid badge + text
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=LABEL_BG_COLOR)
        draw.text(
            (tag_x1 + LABEL_PADDING, tag_y1 + LABEL_PADDING),
            label, fill=LABEL_TEXT_COLOR, font=font,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()
