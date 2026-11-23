"""Shared look, sound and helpers for every explainer video.

Visual identity: "light inside a dark machine". A deep blueprint background with a faint
dot grid; data is light (glows, flowing particles). 3Blue1Brown colors keep fixed meanings:
teal = keys, orange = values, yellow = questions and the KV cache, red = the bug, green = the fix.

Rules for scenes that use this file:
- One caption on screen at a time (`say`). Each caption is also spoken (narration).
- Short sentences, simple words. One word for each thing; never swap in a synonym.
- No LaTeX on this machine: use `T` (text) and `M` (monospace numbers) only.
  Do not use MathTex, Tex, DecimalNumber, Integer, Brace.get_text or chart labels.
- Glow only the object that matters right now. Keep the rest calm.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
    NORMAL,
    RIGHT,
    UL,
    UP,
    AnimationGroup,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    ImageMobject,
    LaggedStart,
    Line,
    MoveAlongPath,
    Polygon,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    VMobject,
    Write,
    config,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import audio  # noqa: E402

BG = "#0A0E16"
INK = "#ECEFF4"
GREY = "#8A93A6"
DIM = "#3A4458"
BLUE = "#58C4DD"
YELLOW = "#F4D35E"
TEAL = "#5CD0B3"
ORANGE = "#FF9F43"
RED = "#FC6255"
GREEN = "#83C167"
PURPLE = "#B189F5"

FONT = "Avenir Next"
DISPLAY = "Futura"
MONO = "Menlo"

config.background_color = BG

CAPTION_WIDTH = 12.2
SECONDS_PER_WORD = 0.30


def T(s, size=36, color=INK, weight=NORMAL, font=FONT, **kw):
    return Text(s, font=font, font_size=size, color=color, weight=weight, **kw)


def M(s, size=30, color=INK, **kw):
    return Text(s, font=MONO, font_size=size, color=color, **kw)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


# ---------------------------------------------------------------- background

def background_image() -> Path:
    """Blueprint backdrop: radial gradient, faint dot grid, vignette, light grain."""
    out = audio.CACHE / "bg_v1.png"
    if out.exists():
        return out
    from PIL import Image

    w, h = 1920, 1080
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x - w * 0.5) / (w * 0.62)) ** 2 + ((y - h * 0.44) / (h * 0.62)) ** 2)
    t = np.clip(r, 0, 1.25)[..., None] / 1.25
    center = np.array([0x13, 0x1C, 0x30], np.float32)
    edge = np.array([0x05, 0x07, 0x0C], np.float32)
    img = center * (1 - t) + edge * t
    grid = ((x % 40) < 2) & ((y % 40) < 2)
    img[grid] = img[grid] * 0.75 + np.array([0x6E, 0x80, 0xA6], np.float32) * 0.25 * (1 - t[grid, 0] * 0.7)[:, None]
    rng = np.random.default_rng(3)
    img += rng.normal(0, 1.6, img.shape[:2])[..., None]
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    return out


# ---------------------------------------------------------------- shapes

def glow(m, color, layers=8, width=28, opacity=0.05):
    """Soft light around a shape: stacked wide, faint strokes behind it."""
    g = VGroup()
    for i in range(layers, 0, -1):
        c = m.copy()
        c.set_fill(opacity=0).set_stroke(color, width=width * i / layers, opacity=opacity)
        g.add(c)
    g.set_z_index(-1)
    return g


def light_dot(color=YELLOW, radius=0.055):
    core = Dot(radius=radius, color=INK)
    halo = VGroup(*[Dot(radius=radius * k, color=color, fill_opacity=0.5 / k) for k in (1.8, 3.0, 4.5)])
    return VGroup(halo, core)


def token_card(word, color=BLUE, width=None, height=0.8, size=32):
    label = T(word, size)
    width = width or max(1.1, label.width + 0.5)
    box = RoundedRectangle(
        corner_radius=0.15,
        width=width,
        height=height,
        stroke_color=color,
        stroke_width=3,
        fill_color=color,
        fill_opacity=0.22,
    )
    box.set_sheen(0.35, UL)
    return VGroup(box, label.move_to(box))


def vec(values, color=TEAL, cell=0.62, size=24):
    """A vector drawn as a row of cells. Cells grow to fit the widest number."""
    widest = max(M(fmt(v), size).width for v in values)
    cell = max(cell, widest + 0.22)
    cells = VGroup()
    for v in values:
        sq = Square(cell, stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=0.14)
        sq.set_sheen(0.25, UL)
        num = M(fmt(v), size)
        cells.add(VGroup(sq, num.move_to(sq)))
    return cells.arrange(RIGHT, buff=0)


def kv_pair(scale=1.0):
    """A small key cell (teal) and value cell (orange)."""
    k = VGroup(Square(0.36, stroke_color=TEAL, fill_color=TEAL, fill_opacity=0.4), T("k", 18))
    v = VGroup(Square(0.36, stroke_color=ORANGE, fill_color=ORANGE, fill_opacity=0.4), T("v", 18))
    k[1].move_to(k[0])
    v[1].move_to(v[0])
    return VGroup(k, v).arrange(RIGHT, buff=0.04).scale(scale)


def notebook(width=4.2, height=2.4, title="KV cache", color=YELLOW):
    """The KV cache drawn as a glowing notebook with ruled lines and rings."""
    shadow = RoundedRectangle(corner_radius=0.18, width=width, height=height, stroke_width=0,
                              fill_color="#000000", fill_opacity=0.45).shift(DOWN * 0.08 + RIGHT * 0.08)
    body = RoundedRectangle(corner_radius=0.18, width=width, height=height, stroke_color=color,
                            stroke_width=3, fill_color="#1A1708", fill_opacity=0.9)
    lines = VGroup(*[
        Line(body.get_left() + RIGHT * 0.45, body.get_right() + LEFT * 0.2, stroke_color=color,
             stroke_width=1, stroke_opacity=0.18).shift(UP * (height / 2 - 0.35 - i * 0.32))
        for i in range(int((height - 0.4) / 0.32))
    ])
    rings = VGroup(*[Circle(0.07, stroke_color=color, stroke_width=2.5) for _ in range(int(height / 0.4))])
    rings.arrange(DOWN, buff=0.22).move_to(body.get_left() + RIGHT * 0.22)
    label = T(title, 26, color, weight=BOLD).next_to(body, UP, buff=0.12)
    return VGroup(VGroup(shadow, glow(body, color, opacity=0.035), body, lines, rings), label)


def check_mark(color=GREEN, size=0.5):
    m = VMobject(stroke_color=color, stroke_width=9)
    m.set_points_as_corners([LEFT * 0.5 * size, DOWN * 0.45 * size, UP * 0.6 * size + RIGHT * 0.7 * size])
    return m


def x_mark(color=RED, size=0.45):
    a = Line(UL * size / 1.4, -UL * size / 1.4, stroke_color=color, stroke_width=9)
    b = Line((UP + RIGHT) * size / 1.4, (DOWN + LEFT) * size / 1.4, stroke_color=color, stroke_width=9)
    return VGroup(a, b)


def chip(text, color=BLUE, size=22, pad=0.3, height=0.5):
    label = T(text, size)
    box = RoundedRectangle(corner_radius=0.1, width=label.width + pad, height=height,
                           stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=0.2)
    box.set_sheen(0.3, UL)
    return VGroup(box, label.move_to(box))


def panel(width, height, color=GREY, title=None, title_size=24):
    box = RoundedRectangle(corner_radius=0.2, width=width, height=height, stroke_color=color,
                           stroke_width=2.5, fill_color="#0C121D", fill_opacity=0.85)
    if title is None:
        return VGroup(box)
    label = T(title, title_size, color, weight=BOLD).next_to(box.get_top(), DOWN, 0.15)
    return VGroup(box, label)


def floor_3d(width, height, color, depth=(0.32, 0.18), top=False, label=None, label_size=22):
    """One floor of a tower with a lit front, a darker side, and an optional top."""
    front = RoundedRectangle(corner_radius=0.04, width=width, height=height, stroke_color=color,
                             stroke_width=2, fill_color=color, fill_opacity=0.2)
    front.set_sheen(0.3, UL)
    d = np.array([depth[0], depth[1], 0])
    br, tr, tl = front.get_corner(DOWN + RIGHT), front.get_corner(UP + RIGHT), front.get_corner(UP + LEFT)
    side = Polygon(br, tr, tr + d, br + d, stroke_color=color, stroke_width=1.5,
                   fill_color=color, fill_opacity=0.08)
    parts = VGroup(side, front)
    if top:
        parts.add(Polygon(tl, tr, tr + d, tl + d, stroke_color=color, stroke_width=1.5,
                          fill_color=color, fill_opacity=0.14))
    if label:
        parts.add(T(label, label_size).move_to(front))
    return parts


# ---------------------------------------------------------------- scene base

class Explainer(Scene):
    """Base scene: background, chapter title, narrated captions, sound effects."""

    def setup(self):
        self.camera.background_color = BG
        self._caption = None
        self._chapter = None
        self._speech_end = 0.0
        self._bg = ImageMobject(str(background_image())).set_z_index(-100)
        self._bg.height = config.frame_height
        self.add(self._bg)
        # Every scene gets an audio track, so scenes join cleanly.
        self.add_sound(str(audio.silence(0.1)))

    # -- sound
    def wait_for_voice(self):
        gap = self._speech_end - self.renderer.time
        if gap > 0.05:
            self.wait(gap)

    def voice(self, text, speak=None):
        """Start narration now. Returns its length in seconds."""
        self.wait_for_voice()
        path, spoken = audio.narrate(text, speak)
        self.add_sound(str(path))
        self._speech_end = self.renderer.time + spoken + 0.2
        return spoken

    def sfx(self, name, gain=-6.0, offset=0.0):
        self.add_sound(str(audio.sfx(name)), time_offset=offset, gain=gain)

    # -- chapter card: big outlined number, title, underline; then title moves to the corner
    def chapter(self, title):
        num, name = title.split(" · ", 1)
        number = Text(num, font=DISPLAY, weight=BOLD, font_size=230)
        number.set_fill(opacity=0).set_stroke(YELLOW, width=2.5, opacity=0.55)
        heading = Text(name, font=DISPLAY, weight=BOLD, font_size=58, color=INK)
        group = VGroup(number, heading).arrange(RIGHT, buff=0.55, aligned_edge=DOWN).move_to(UP * 0.3)
        rule = Line(heading.get_corner(DOWN + LEFT), heading.get_corner(DOWN + RIGHT), color=YELLOW,
                    stroke_width=4).shift(DOWN * 0.22)
        self.wait_for_voice()
        self.sfx("whoosh", gain=-10)
        self.play(Write(number, run_time=1.1), FadeIn(heading, shift=RIGHT * 0.4, run_time=0.9),
                  Create(rule, run_time=0.9))
        self.wait(0.7)
        corner = T(title, 22, YELLOW, weight=BOLD).to_corner(UL, buff=0.35)
        self.play(FadeOut(number, scale=1.15), FadeOut(rule), FadeTransform(heading, corner), run_time=0.8)
        self._chapter = corner

    # -- narrated caption on a dark backing
    def say(self, text, *anims, run_time=0.8, wait=None, speak=None, voice=True):
        """Swap the caption, speak it, play `anims` at the same time, then hold.

        With `wait` given, the scene moves on while the voice is still talking; the next
        caption waits for the voice to finish, so two lines never overlap.
        """
        label = T(text, 30)
        if label.width > CAPTION_WIDTH:
            label.scale_to_fit_width(CAPTION_WIDTH)
        backing = RoundedRectangle(corner_radius=0.22, width=label.width + 0.7, height=label.height + 0.34,
                                   stroke_width=0, fill_color="#05080E", fill_opacity=0.72)
        new = VGroup(backing, label.move_to(backing)).to_edge(DOWN, buff=0.3)
        swap = [FadeIn(new, shift=UP * 0.12)]
        if self._caption is not None:
            swap.append(FadeOut(self._caption, shift=UP * 0.12))
        spoken = self.voice(text, speak) if voice else 0.0
        self.play(AnimationGroup(*swap, *anims), run_time=run_time)
        self._caption = new
        if wait is None:
            reading = max(1.5, SECONDS_PER_WORD * len(text.split()) + 0.8)
            wait = max(reading, spoken + 0.3 - run_time)
        if wait > 0:
            self.wait(wait)

    # -- data as light, moving along paths
    def flow(self, *paths, color=YELLOW, n=6, run_time=1.4, lag=0.12, sound=True):
        dots = []
        anims = []
        for path in paths:
            for _ in range(n):
                d = light_dot(color).move_to(path.get_start())
                dots.append(d)
                anims.append(MoveAlongPath(d, path))
        if sound:
            self.sfx("swell", gain=-14)
        self.add(*dots)
        self.play(LaggedStart(*anims, lag_ratio=lag), run_time=run_time)
        self.remove(*dots)

    def clean(self, *keep, run_time=0.6):
        """Fade out everything except the background, chapter, caption and `keep`."""
        keep_set = {self._bg, self._chapter, self._caption, *keep}
        gone = [m for m in self.mobjects if m not in keep_set]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    def end(self, hold=0.6):
        self.wait_for_voice()
        self.wait(hold)
        gone = [m for m in self.mobjects if m is not self._bg]
        self.play(*[FadeOut(m) for m in gone], run_time=0.6)
