"""
Game-style frame for the CHAIN 3D tasks (PipeFit, FluidNet, Domino).

plotly renders only the 3D scene (transparent background); this module
composes the final PNG around it with PIL:

  * a themed gradient background with a soft glow behind the board
  * a top title bar: game name, level / step caption, result badge
  * right-hand HUD cards: STATUS (with progress bars), INVENTORY (item
    counters) and LEGEND (a drawn icon for every piece type)
  * a bottom hint strip with the one-line rules

Everything a player (or a model) needs is still printed as text; the icons
only make the pieces recognisable at a glance.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

RGB = Tuple[int, int, int]
Icon = Callable[[ImageDraw.ImageDraw, Tuple[int, int, int, int]], None]

FONT_REG = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
    except OSError:
        return ImageFont.load_default()


def hex_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def shade(c: RGB, f: float) -> RGB:
    """f < 1 darkens, f > 1 lightens (towards white)."""
    if f <= 1:
        return tuple(max(0, int(v * f)) for v in c)  # type: ignore
    return tuple(min(255, int(v + (255 - v) * (f - 1))) for v in c)  # type: ignore


# ---------------------------------------------------------------------------
# Theme / HUD data
# ---------------------------------------------------------------------------

@dataclass
class Theme:
    name: str
    bg_top: str
    bg_bottom: str
    glow: str
    accent: str
    card_fill: Tuple[int, int, int, int] = (12, 16, 30, 205)
    text: str = "#EEF2F8"
    muted: str = "#A9B4C6"


@dataclass
class Stat:
    icon: Optional[Icon]
    label: str
    value: str
    progress: Optional[float] = None      # 0..1 draws a bar
    color: Optional[str] = None           # bar / value colour


@dataclass
class Item:
    icon: Optional[Icon]
    label: str
    left: int
    total: int


@dataclass
class LegendEntry:
    icon: Optional[Icon]
    name: str
    desc: str


@dataclass
class Hud:
    title: str
    caption: str
    result: str                     # "PLAYING" / "WIN" / "LOSE" / phase text
    stats: List[Stat] = field(default_factory=list)
    inventory: List[Item] = field(default_factory=list)
    legend: List[LegendEntry] = field(default_factory=list)
    hint: str = ""
    inventory_title: str = "INVENTORY"


RESULT_COLORS = {
    "WIN": "#2BD07A",
    "LOSE": "#FF4D5E",
    "PLAYING": "#4DA3FF",
}


# ---------------------------------------------------------------------------
# Icon primitives (drawn into a square box)
# ---------------------------------------------------------------------------

def _iso_box(d: ImageDraw.ImageDraw, cx: float, base_y: float, w: float,
             h: float, color: RGB, outline: RGB = (20, 20, 28)) -> None:
    """Isometric box: footprint width w, height h, bottom vertex at base_y."""
    dx, dy = w / 2, w / 4
    top = [(cx, base_y - h - 2 * dy), (cx + dx, base_y - h - dy),
           (cx, base_y - h), (cx - dx, base_y - h - dy)]
    left = [(cx - dx, base_y - h - dy), (cx, base_y - h), (cx, base_y),
            (cx - dx, base_y - dy)]
    right = [(cx, base_y - h), (cx + dx, base_y - h - dy),
             (cx + dx, base_y - dy), (cx, base_y)]
    d.polygon(left, fill=shade(color, 0.78), outline=outline)
    d.polygon(right, fill=shade(color, 0.58), outline=outline)
    d.polygon(top, fill=shade(color, 1.12), outline=outline)


def icon_cube(color: str, scale: float = 0.62) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        s = (x1 - x0) * scale
        _iso_box(d, (x0 + x1) / 2, y1 - (y1 - y0 - s * 1.2) / 2, s, s * 0.55,
                 hex_rgb(color))
    return draw


def icon_slab(color: str, mark: Optional[str] = None,
              mark_color: str = "#FFFFFF") -> Icon:
    """Tall standing slab (domino-like) with an optional top mark."""
    def draw(d, box):
        x0, y0, x1, y1 = box
        w = (x1 - x0)
        cx, base = (x0 + x1) / 2, y1 - 3
        sw, sh = w * 0.42, (y1 - y0) * 0.62
        _iso_box(d, cx, base, sw, sh, hex_rgb(color))
        top_y = base - sh - sw / 4
        mc = hex_rgb(mark_color)
        r = sw * 0.22
        if mark == "x":
            d.line([(cx - r, top_y - r / 2), (cx + r, top_y + r / 2)], fill=mc, width=3)
            d.line([(cx - r, top_y + r / 2), (cx + r, top_y - r / 2)], fill=mc, width=3)
        elif mark == "star":
            _star(d, cx, top_y, r * 1.25, mc)
        elif mark == "cross":
            d.line([(cx - r * 1.5, top_y), (cx + r * 1.5, top_y)], fill=mc, width=4)
            d.line([(cx, top_y - r * 0.8), (cx, top_y + r * 0.8)], fill=mc, width=4)
        elif mark == "dot":
            d.ellipse([cx - r, top_y - r * 0.55, cx + r, top_y + r * 0.55],
                      fill=mc)
        elif mark == "pips":
            for py in (0.35, 0.7):
                for px in (-0.12, 0.12):
                    yy = base - sh * (1 - py) - sw * 0.06
                    d.ellipse([cx - sw * 0.25 + px * sw - 2, yy - 2,
                               cx - sw * 0.25 + px * sw + 2, yy + 2], fill=(25, 20, 15))
    return draw


def icon_flat(color: str) -> Icon:
    """A toppled piece: low slab with a fall arrow."""
    def draw(d, box):
        x0, y0, x1, y1 = box
        w = x1 - x0
        cx, base = (x0 + x1) / 2, y1 - w * 0.25
        _iso_box(d, cx, base, w * 0.78, w * 0.14, hex_rgb(color))
        ty = base - w * 0.14 - w * 0.2
        d.line([(cx - w * 0.18, ty + w * 0.05), (cx + w * 0.16, ty - w * 0.05)],
               fill=(255, 255, 255), width=3)
        d.polygon([(cx + w * 0.22, ty - w * 0.07), (cx + w * 0.08, ty - w * 0.09),
                   (cx + w * 0.14, ty + w * 0.02)], fill=(255, 255, 255))
    return draw


def icon_steps(color: str) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        w = x1 - x0
        cx, base = (x0 + x1) / 2, y1 - 4
        c = hex_rgb(color)
        for k, (s, h) in enumerate(((0.80, 0.16), (0.56, 0.32), (0.32, 0.48))):
            _iso_box(d, cx, base - (0 if k == 0 else 0), w * s, w * h,
                     c if k % 2 == 0 else shade(c, 0.88))
    return draw


def icon_gem(color: str, ring: str = "#D9C280") -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = (x1 - x0) * 0.36
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=hex_rgb(ring),
                  outline=(30, 30, 30))
        r2 = r * 0.66
        c = hex_rgb(color)
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=c)
        d.ellipse([cx - r2 * 0.55, cy - r2 * 0.75, cx - r2 * 0.05, cy - r2 * 0.25],
                  fill=shade(c, 1.6))
    return draw


def icon_sphere(color: str) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = (x1 - x0) * 0.34
        c = hex_rgb(color)
        for k in range(8, 0, -1):
            rr = r * k / 8
            off = r * (1 - k / 8) * 0.35
            d.ellipse([cx - rr - off, cy - rr - off, cx + rr - off, cy + rr - off],
                      fill=shade(c, 0.7 + 0.12 * (8 - k) / 2))
    return draw


def icon_pipe(shape: str, color: str = "#3A4556",
              ring: Optional[str] = "#B8922A") -> Icon:
    """I / L / T pipe glyph (top view) with brass flanges at the ends."""
    def draw(d, box):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = (x1 - x0) * 0.40
        c = hex_rgb(color)
        arms = {"I": [(-1, 0), (1, 0)], "L": [(-1, 0), (0, -1)],
                "T": [(-1, 0), (1, 0), (0, 1)], "bar": [(-1, 0), (1, 0)]}[shape]
        wdt = max(6, int((x1 - x0) * 0.2))
        for ax, ay in arms:
            d.line([(cx, cy), (cx + ax * r, cy + ay * r)], fill=c, width=wdt)
        d.ellipse([cx - wdt * 0.6, cy - wdt * 0.6, cx + wdt * 0.6, cy + wdt * 0.6], fill=c)
        if ring:
            rc = hex_rgb(ring)
            for ax, ay in arms:
                ex, ey = cx + ax * r * 0.92, cy + ay * r * 0.92
                if ax:
                    d.rectangle([ex - 2, ey - wdt * 0.75, ex + 2, ey + wdt * 0.75], fill=rc)
                else:
                    d.rectangle([ex - wdt * 0.75, ey - 2, ex + wdt * 0.75, ey + 2], fill=rc)
    return draw


def icon_pillar(color: str, cells: int) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        w = x1 - x0
        cx, base = (x0 + x1) / 2, y1 - 3
        h = (y1 - y0) * (0.34 if cells == 1 else 0.62)
        _iso_box(d, cx, base, w * 0.42, h, hex_rgb(color))
        if cells == 2:
            my = base - h / 2 - w * 0.05
            d.line([(cx - w * 0.21, my - w * 0.05), (cx, my + w * 0.05),
                    (cx + w * 0.21, my - w * 0.05)], fill=(40, 30, 20), width=2)
    return draw


def icon_block_x(color: str) -> Icon:
    def draw(d, box):
        icon_cube(color, 0.66)(d, box)
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        s = (x1 - x0) * 0.66
        ty = y1 - (y1 - y0 - s * 1.2) / 2 - s * 0.55 - s / 4
        r = s * 0.2
        d.line([(cx - r, ty - r / 2), (cx + r, ty + r / 2)], fill="white", width=3)
        d.line([(cx - r, ty + r / 2), (cx + r, ty - r / 2)], fill="white", width=3)
    return draw


def icon_valve(color: str = "#F2B315", open_: bool = False) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        if open_:
            # Matches the scene: a fluid-coloured ball with an amber disc.
            icon_sphere("#1E5FB8")(d, box)
            cx = (x0 + x1) / 2
            r = (x1 - x0) * 0.2
            top = (y0 + y1) / 2 - (x1 - x0) * 0.36
            d.ellipse([cx - r, top - r * 0.35, cx + r, top + r * 0.35],
                      fill=(255, 215, 90), outline=(90, 70, 10))
            return
        icon_cube(color, 0.56)(d, box)
        cx = (x0 + x1) / 2
        s = (x1 - x0) * 0.56
        ty = y1 - (y1 - y0 - s * 1.2) / 2 - s * 0.55 - s / 4 - 6
        r = s * 0.32
        d.line([(cx, ty + 6), (cx, ty)], fill=(26, 31, 38), width=3)
        d.ellipse([cx - r, ty - r / 2, cx + r, ty + r / 2], outline=(26, 31, 38), width=3)
        d.line([(cx - r, ty), (cx + r, ty)], fill=(26, 31, 38), width=2)
    return draw


def icon_target(color: str) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        w = x1 - x0
        cx, base = (x0 + x1) / 2, y1 - w * 0.14
        s = w * 0.6
        c = hex_rgb(color)
        dx, dy = s / 2, s / 4
        pts = [(cx, base - s * 0.55 - 2 * dy), (cx + dx, base - s * 0.55 - dy),
               (cx, base - s * 0.55), (cx - dx, base - s * 0.55 - dy)]
        d.polygon([(cx - dx, base - dy - s * 0.55), (cx, base - s * 0.55),
                   (cx + dx, base - dy - s * 0.55), (cx + dx, base - dy),
                   (cx, base), (cx - dx, base - dy)], fill=shade(c, 1.75))
        for a, b in [((cx - dx, base - dy), (cx, base)), ((cx, base), (cx + dx, base - dy)),
                     ((cx, base), (cx, base - s * 0.55))]:
            d.line([a, b], fill=c, width=2)
        d.polygon(pts, outline=c)
        for k in range(4):
            d.line([(pts[k][0], pts[k][1]), (pts[k][0], pts[k][1] + s * 0.55)],
                   fill=c, width=2)
    return draw


def icon_ring(color: str) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = (x1 - x0) * 0.36
        d.ellipse([cx - r, cy - r * 0.55, cx + r, cy + r * 0.55],
                  outline=hex_rgb(color), width=4)
    return draw


def icon_arrow(color: str) -> Icon:
    def draw(d, box):
        x0, y0, x1, y1 = box
        cy = (y0 + y1) / 2
        c = hex_rgb(color)
        d.line([(x0 + 6, cy), (x1 - 14, cy)], fill=c, width=5)
        d.polygon([(x1 - 4, cy), (x1 - 18, cy - 9), (x1 - 18, cy + 9)], fill=c)
    return draw


def _star(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill) -> None:
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.42
        a = -math.pi / 2 + k * math.pi / 5
        pts.append((cx + rr * math.cos(a), cy + rr * 0.6 * math.sin(a)))
    d.polygon(pts, fill=fill)


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def _gradient(size: Tuple[int, int], top: RGB, bottom: RGB) -> Image.Image:
    w, h = size
    col = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        col.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t)
                                   for i in range(3)))
    return col.resize((w, h))


def _overlay(canvas: Image.Image, layer: Image.Image) -> None:
    """Blend an RGBA layer onto the RGB canvas."""
    canvas.paste(layer, (0, 0), layer)


def _rounded(img: Image.Image, box, radius: int, fill, outline=None,
             width: int = 2, shadow: bool = True) -> None:
    """Rounded card with a soft drop shadow, blended onto img."""
    x0, y0, x1, y1 = [int(v) for v in box]
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([x0 + 4, y0 + 8, x1 + 4, y1 + 8],
                                             radius, fill=(0, 0, 0, 120))
        _overlay(img, sh.filter(ImageFilter.GaussianBlur(10)))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([x0, y0, x1, y1], radius, fill=fill,
                                            outline=outline, width=width)
    _overlay(img, layer)


def _card_header(d: ImageDraw.ImageDraw, x: int, y: int, w: int, text: str,
                 theme: Theme) -> int:
    d.rounded_rectangle([x + 18, y + 18, x + 24, y + 42], 3, fill=theme.accent)
    d.text((x + 34, y + 16), text, font=font(22, True), fill=theme.text)
    return y + 56


def compose(scene_png: bytes, hud: Hud, theme: Theme,
            size: Tuple[int, int], scene_box: Tuple[int, int, int, int],
            ) -> Image.Image:
    """Build the final frame around a transparent scene render."""
    W, H = size
    canvas = _gradient(size, hex_rgb(theme.bg_top), hex_rgb(theme.bg_bottom))

    # Soft glow behind the board.
    sx0, sy0, sx1, sy1 = scene_box
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gc = hex_rgb(theme.glow)
    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) * 0.56
    rx, ry = (sx1 - sx0) * 0.42, (sy1 - sy0) * 0.36
    ImageDraw.Draw(glow).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=gc + (110,))
    _overlay(canvas, glow.filter(ImageFilter.GaussianBlur(120)))

    scene = Image.open(io.BytesIO(scene_png)).convert("RGBA")
    if scene.size != (sx1 - sx0, sy1 - sy0):
        scene = scene.resize((sx1 - sx0, sy1 - sy0), Image.LANCZOS)
    canvas.paste(scene, (sx0, sy0), scene)

    # On an RGB image, "RGBA" mode blends translucent fills.
    d = ImageDraw.Draw(canvas, "RGBA")
    accent = hex_rgb(theme.accent)

    # ---- title bar ----
    bar_h = 104
    bar = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(bar).rectangle([0, 0, W, bar_h], fill=(6, 8, 16, 170))
    _overlay(canvas, bar)
    d.rectangle([0, bar_h - 4, W, bar_h], fill=accent)
    d.text((36, 18), hud.title, font=font(46, True), fill=theme.text)
    tw = d.textlength(hud.title, font=font(46, True))
    d.text((36 + tw + 26, 36), hud.caption, font=font(24), fill=theme.muted)
    res = hud.result.upper()
    rc = RESULT_COLORS.get(res, theme.accent)
    rf = font(26, True)
    rw = d.textlength(res, font=rf)
    bx1, by0 = W - 36, 26
    bx0 = bx1 - rw - 56
    d.rounded_rectangle([bx0, by0, bx1, by0 + 52], 26, fill=hex_rgb(rc))
    d.ellipse([bx0 + 16, by0 + 19, bx0 + 30, by0 + 33], fill=(255, 255, 255))
    d.text((bx0 + 40, by0 + 10), res, font=rf, fill=(255, 255, 255))

    # ---- right column cards ----
    col_x0, col_x1 = W - 470, W - 30
    y = bar_h + 26
    fill = theme.card_fill
    border = accent + (200,)

    if hud.stats:
        h = 64 + 62 * len(hud.stats)
        _rounded(canvas, (col_x0, y, col_x1, y + h), 22, fill, border)
        yy = _card_header(d, col_x0, y, col_x1 - col_x0, "STATUS", theme)
        for st in hud.stats:
            if st.icon:
                st.icon(d, (col_x0 + 20, yy, col_x0 + 62, yy + 42))
            d.text((col_x0 + 74, yy + 2), st.label, font=font(20), fill=theme.muted)
            vf = font(22, True)
            vc = hex_rgb(st.color) if st.color else hex_rgb(theme.text)
            d.text((col_x1 - 22 - d.textlength(st.value, font=vf), yy), st.value,
                   font=vf, fill=vc)
            if st.progress is not None:
                px0, px1, py = col_x0 + 74, col_x1 - 22, yy + 32
                d.rounded_rectangle([px0, py, px1, py + 10], 5, fill=(255, 255, 255, 40))
                pw = int((px1 - px0) * max(0.0, min(1.0, st.progress)))
                if pw > 0:
                    d.rounded_rectangle([px0, py, px0 + max(pw, 10), py + 10], 5,
                                        fill=vc)
            yy += 62
        y += h + 22

    if hud.inventory:
        h = 64 + 60 * len(hud.inventory)
        _rounded(canvas, (col_x0, y, col_x1, y + h), 22, fill, border)
        yy = _card_header(d, col_x0, y, col_x1 - col_x0, hud.inventory_title, theme)
        cap = "left / total"
        d.text((col_x1 - 22 - d.textlength(cap, font=font(16)), y + 22), cap,
               font=font(16), fill=theme.muted)
        for it in hud.inventory:
            if it.icon:
                it.icon(d, (col_x0 + 20, yy - 4, col_x0 + 66, yy + 42))
            d.text((col_x0 + 78, yy + 6), it.label, font=font(21, True), fill=theme.text)
            chip = f"{it.left} / {it.total}"
            cf = font(21, True)
            cw = d.textlength(chip, font=cf)
            cx1 = col_x1 - 22
            done = it.left == 0
            d.rounded_rectangle([cx1 - cw - 30, yy + 2, cx1, yy + 38], 18,
                                fill=(255, 255, 255, 30) if done else accent + (230,))
            d.text((cx1 - cw - 15, yy + 7), chip, font=cf,
                   fill=hex_rgb(theme.muted) if done else (255, 255, 255))
            yy += 60
        y += h + 22

    if hud.legend:
        row = 66
        h = 64 + row * len(hud.legend)
        h = min(h, H - y - 30)
        _rounded(canvas, (col_x0, y, col_x1, y + h), 22, fill, border)
        yy = _card_header(d, col_x0, y, col_x1 - col_x0, "LEGEND", theme)
        for le in hud.legend:
            if yy + row > y + h + 6:
                break
            if le.icon:
                le.icon(d, (col_x0 + 16, yy - 4, col_x0 + 70, yy + 50))
            d.text((col_x0 + 80, yy), le.name, font=font(20, True), fill=theme.text)
            d.text((col_x0 + 80, yy + 26), le.desc, font=font(17), fill=theme.muted)
            yy += row

    # ---- hint strip ----
    if hud.hint:
        hf = font(21)
        hw = d.textlength(hud.hint, font=hf)
        hx0 = max(30, (sx0 + sx1) / 2 - hw / 2 - 26)
        _rounded(canvas, (hx0, H - 74, hx0 + hw + 52, H - 24), 25,
                 (6, 8, 16, 190), accent + (160,), width=2, shadow=False)
        d.text((hx0 + 26, H - 63), hud.hint, font=hf, fill=theme.text)

    return canvas
