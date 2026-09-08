"""Render the application icon set from scratch, at full resolution.

    python packaging/make_icons.py

Writes packaging/icon_master.png (1024), icon.ico (Windows) and icon.icns
(macOS), plus icon_preview.png -- a sheet showing every size as it will
actually be displayed, which is the only honest way to judge an icon.

Why this script exists
----------------------
The previous icon set was derived from a single 512x512 raster whose artwork
covered 6.3% of the canvas: the wordmark was 22% of the frame's height, on a
fully transparent background. Scaled to the sizes that actually get shown --
16 and 32 px in a taskbar or file list -- the lettering came out three to
seven pixels tall, which reads as a smudge, and the dark glyphs vanished
against a dark taskbar because nothing sat behind them.

Rendering procedurally fixes both halves of that. Everything is drawn at 4x
and downsampled, so edges stay clean at any size; and the artwork adapts to
the size it is being drawn for, which is a normal and well-supported thing for
an .ico to do:

  <= 48 px   the scattering mark alone -- a horizon line and a Bragg rod,
             the shape of a grazing-incidence pattern. Legible at 16 px,
             where any wordmark would not be.
  >= 64 px   the mark plus the "p-GIXS" wordmark, which needs roughly 12 px
             of glyph height before it stops being a grey band.

Both sit on an opaque rounded square, so the icon has the same presence on a
light desktop and a dark one.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent

MASTER = 1024
SS = 4                                   # supersampling factor while drawing
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]

BG_TOP = (28, 58, 107)                   # deep beamline blue
BG_BOTTOM = (17, 34, 66)
ACCENT = (110, 168, 235)                 # the blue already used in the wordmark
WHITE = (255, 255, 255)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/seguibl.ttf",      # Segoe UI Black
    "C:/Windows/Fonts/ariblk.ttf",       # Arial Black
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(px: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def rounded_background(size: int) -> Image.Image:
    """Opaque rounded square with a vertical gradient."""

    gradient = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / max(size - 1, 1)
        gradient.putpixel((0, y), tuple(
            int(round(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t)) for i in range(3)
        ))
    image = gradient.resize((size, size), Image.NEAREST).convert("RGBA")

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * 0.22), fill=255
    )
    image.putalpha(mask)
    return image


def draw_scattering_mark(draw: ImageDraw.ImageDraw, size: int, cx: float, cy: float,
                         scale: float) -> None:
    """A grazing-incidence pattern reduced to its two recognisable parts: the
    horizon band, and a Bragg rod standing on it with its symmetric in-plane
    pair."""

    w = size * scale
    line_w = max(2, int(size * 0.026))

    # Horizon.
    draw.line([(cx - w * 0.50, cy), (cx + w * 0.50, cy)], fill=ACCENT + (255,), width=line_w)

    # Out-of-plane rod: a tapering vertical streak above the centre.
    rod_h = w * 0.50
    rod_w = w * 0.17
    for i in range(40):
        t = i / 39.0
        half = rod_w * (1.0 - 0.30 * t) * 0.5
        y = cy - rod_h * t
        alpha = int(255 * (1.0 - 0.35 * t))
        draw.ellipse((cx - half, y - half * 0.85, cx + half, y + half * 0.85),
                     fill=WHITE + (alpha,))

    # In-plane reflections either side.
    for sign in (-1, 1):
        px = cx + sign * w * 0.31
        r = w * 0.085
        draw.ellipse((px - r, cy - r * 1.35, px + r, cy + r * 1.35), fill=WHITE + (235,))


def render(size: int, with_wordmark: bool) -> Image.Image:
    """One icon, drawn at SS x `size` and downsampled."""

    s = size * SS
    canvas = rounded_background(s)
    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if with_wordmark:
        draw_scattering_mark(draw, s, cx=s * 0.5, cy=s * 0.50, scale=0.70)

        text = "p-GIXS"
        font_px = int(s * 0.235)
        font = load_font(font_px)
        # Shrink to fit the width with a margin, rather than trusting one ratio
        # across every font this might run with.
        while font_px > 8:
            font = load_font(font_px)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= s * 0.84:
                break
            font_px -= 2
        box = draw.textbbox((0, 0), text, font=font)
        tx = (s - (box[2] - box[0])) / 2 - box[0]
        ty = s * 0.655 - box[1]
        # Offset shadow in the accent blue, as the original wordmark had.
        draw.text((tx + s * 0.008, ty + s * 0.008), text, font=font, fill=ACCENT + (255,))
        draw.text((tx, ty), text, font=font, fill=WHITE + (255,))
    else:
        # Small sizes: the mark alone, centred and much larger.
        draw_scattering_mark(draw, s, cx=s * 0.5, cy=s * 0.63, scale=1.02)

    canvas = Image.alpha_composite(canvas, layer)
    if SS > 1:
        canvas = canvas.resize((size, size), Image.LANCZOS)
    # A touch of sharpening recovers the crispness LANCZOS softens at tiny sizes.
    if size <= 48:
        canvas = canvas.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=0))
    return canvas


def icon_for(size: int) -> Image.Image:
    return render(size, with_wordmark=size >= 64)


def preview_sheet(sizes: list[int]) -> Image.Image:
    pad = 16
    height = max(sizes) + pad * 3 + 20
    width = sum(s + pad for s in sizes) + pad
    # Split light/dark so the icon can be judged on both, which is the point of
    # having an opaque background at all.
    sheet = Image.new("RGBA", (width, height * 2), (245, 245, 247, 255))
    ImageDraw.Draw(sheet).rectangle((0, height, width, height * 2), fill=(32, 32, 36, 255))

    label_font = load_font(13)
    for band, top in enumerate((0, height)):
        x = pad
        for s in sizes:
            icon = icon_for(s)
            y = top + pad + (max(sizes) - s)
            sheet.alpha_composite(icon, (x, y))
            ImageDraw.Draw(sheet).text(
                (x, top + pad + max(sizes) + 6), f"{s}px", font=label_font,
                fill=(90, 90, 95, 255) if band == 0 else (200, 200, 205, 255))
            x += s + pad
    return sheet


def main() -> int:
    master = icon_for(MASTER)
    master.save(HERE / "icon_master.png")
    print(f"icon_master.png  {MASTER}x{MASTER}")

    ico_images = [icon_for(s) for s in ICO_SIZES]
    ico_images[-1].save(HERE / "icon.ico", format="ICO",
                        sizes=[(s, s) for s in ICO_SIZES], append_images=ico_images[:-1])
    print(f"icon.ico         {ICO_SIZES}")

    icns_images = [icon_for(s) for s in ICNS_SIZES]
    try:
        icns_images[-1].save(HERE / "icon.icns", format="ICNS",
                             sizes=[(s, s) for s in ICNS_SIZES], append_images=icns_images[:-1])
        print(f"icon.icns        {ICNS_SIZES}")
    except Exception as exc:  # noqa: BLE001 - ICNS support varies by Pillow build
        print(f"icon.icns        SKIPPED ({exc})", file=sys.stderr)

    # The in-app window/taskbar icon Qt loads at runtime. It used to point at
    # Intrologo.png, a 399x150 banner, which Qt letterboxes into the square
    # icon slot -- so the running app's icon looked worse than the executable's.
    package_dir = HERE.parent / "pgixs_python"
    # Two files on purpose. Qt reads every size out of the .ico, so the
    # taskbar gets the hand-tuned 16/24/32 px art instead of a 512 px image
    # squeezed down; the .png is the fallback if a platform's Qt build has no
    # ICO plugin.
    ico_images[-1].save(package_dir / "app_icon.ico", format="ICO",
                        sizes=[(s, s) for s in ICO_SIZES], append_images=ico_images[:-1])
    icon_for(512).save(package_dir / "app_icon.png")
    print("app_icon.ico     runtime window icon, all sizes")
    print("app_icon.png     512x512 fallback")

    preview_sheet([16, 24, 32, 48, 64, 128, 256]).convert("RGB").save(HERE / "icon_preview.png")
    print("icon_preview.png  light and dark backgrounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
