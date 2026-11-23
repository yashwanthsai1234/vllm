"""Explainer 01: how a model writes text, the KV cache, and vLLM issue 54711.

Render all scenes and join them (narration + sound effects + music), from explainers/:
    .venv/bin/python common/render.py 01-kv-cache-issue-54711

Facts used here were checked against vLLM main at 6cbea92cb2:
- page size formula: heads x tokens per block x (key + value length) x dtype bytes
- test spec: 2 heads, 16 tokens per block, head size 64, float32 -> 16,384 bytes
- PP split in the issue: 2,16,16,14 = 48 layers; stage 1 holds 2 state (GDN) layers
- fix: change 60234 (2026-10-27) plan reads the name list;
       change 61452 (2026-11-12) names come from the size table.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import (  # noqa: E402
    BOLD,
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    AnimationGroup,
    Arc,
    ArcBetweenPoints,
    Arrow,
    Axes,
    Circumscribe,
    Create,
    CurvedArrow,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    GrowArrow,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    ManimColor,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Square,
    SurroundingRectangle,
    Transform,
    TransformFromCopy,
    VGroup,
    Wiggle,
    Write,
    interpolate_color,
)

from common.style import (  # noqa: E402
    BLUE,
    DIM,
    DISPLAY,
    GREEN,
    GREY,
    INK,
    ORANGE,
    PURPLE,
    RED,
    TEAL,
    YELLOW,
    Explainer,
    M,
    T,
    check_mark,
    chip,
    floor_3d,
    glow,
    kv_pair,
    notebook,
    panel,
    token_card,
    vec,
    x_mark,
)

SCENES = [
    "S01_GuessingGame",
    "S02_Numbers",
    "S03_Tower",
    "S04_Attention",
    "S05_NoCache",
    "S06_KVCache",
    "S07_Blocks",
    "S08_TwoKinds",
    "S09_Split",
    "S10_Bug",
    "S11_Fix",
    "S12_Summary",
]

CHAPTERS = {
    "S01_GuessingGame": "1 · The guessing game",
    "S02_Numbers": "2 · Tokens become numbers",
    "S03_Tower": "3 · The tower of layers",
    "S04_Attention": "4 · Looking back: attention",
    "S05_NoCache": "5 · The tiring way",
    "S06_KVCache": "6 · The smart way: the KV cache",
    "S07_Blocks": "7 · Pages of memory: blocks",
    "S08_TwoKinds": "8 · Two kinds of layers",
    "S09_Split": "9 · Too big for one GPU",
    "S10_Bug": "10 · The bug",
    "S11_Fix": "11 · The fix",
    "S12_Summary": "12 · The whole picture",
}


def halo(m, color, buff=0.1, opacity=0.05):
    """Glow around the bounding box of `m` (use for text and vectors)."""
    return glow(SurroundingRectangle(m, buff=buff, corner_radius=0.1), color, opacity=opacity)


def gpu_chip(size=1.7, color=GREEN, label="GPU"):
    body = RoundedRectangle(corner_radius=0.14, width=size, height=size, stroke_color=color,
                            stroke_width=3, fill_color="#0D1811", fill_opacity=0.95)
    die = RoundedRectangle(corner_radius=0.06, width=size * 0.52, height=size * 0.52, stroke_color=color,
                           stroke_width=2, fill_color=color, fill_opacity=0.22).move_to(body)
    die.set_sheen(0.4, UP + LEFT)
    pins = VGroup()
    for side, along in [(UP, RIGHT), (DOWN, RIGHT), (LEFT, UP), (RIGHT, UP)]:
        for i in range(6):
            off = (-0.36 + i * 0.144) * size
            start = body.get_edge_center(side) + along * off
            pins.add(Line(start, start + side * 0.17, stroke_color=color, stroke_width=3))
    text = T(label, 26, INK, weight=BOLD, font=DISPLAY).move_to(die)
    return VGroup(glow(body, color, opacity=0.035), pins, body, die, text)


class S01_GuessingGame(Explainer):
    def construct(self):
        rng = np.random.default_rng(11)
        drift_words = ["the", "cat", "token", "sat", "mat", "on", "key", "value", "query", "layer",
                       "GPU", "block", "vector", "attention", "cache", "next"]
        drifters = VGroup()
        for w in drift_words:
            m = T(w, int(rng.integers(22, 40)), GREY).set_opacity(0.08)
            m.move_to([rng.uniform(-6.4, 6.4), rng.uniform(-3.4, 3.4), 0])
            v = np.array([rng.uniform(-0.12, 0.12), rng.uniform(0.04, 0.14), 0])
            m.add_updater(lambda mob, dt, v=v: mob.shift(v * dt))
            drifters.add(m)
        title = T("How a model writes text", 64, weight=BOLD, font=DISPLAY)
        sub = T("inference, the KV cache, and vLLM issue 54711", 30, GREY).next_to(title, DOWN, buff=0.4)
        self.voice("How a model writes text.",
                   speak="How a model writes text. Inference, the KV cache, and vLLM issue five four seven one one.")
        self.sfx("swell", gain=-8)
        self.play(FadeIn(drifters), Write(title), run_time=1.6)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait_for_voice()
        self.wait(0.8)
        for m in drifters:
            m.clear_updaters()
        self.play(FadeOut(title), FadeOut(sub), FadeOut(drifters))

        self.chapter(CHAPTERS["S01_GuessingGame"])
        words = ["The", "cat", "sat", "on", "the"]
        cards = VGroup(*[token_card(w) for w in words]).arrange(RIGHT, buff=0.2)
        guess = token_card("?", YELLOW, width=1.1)
        VGroup(cards, guess).arrange(RIGHT, buff=0.2).move_to(UP * 1.6)

        self.sfx("pop", gain=-8)
        self.say(
            "A model is a guessing machine.",
            LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in cards], lag_ratio=0.15),
            run_time=1.4,
        )
        self.say("It reads some words. Then it guesses the next word.", FadeIn(guess))

        model = RoundedRectangle(corner_radius=0.3, width=3.4, height=1.4, stroke_color=PURPLE,
                                 fill_color=PURPLE, fill_opacity=0.18)
        model.set_sheen(0.35, UP + LEFT)
        model_g = VGroup(glow(model, PURPLE), model, T("model", 42, PURPLE, weight=BOLD, font=DISPLAY))
        model_g[2].move_to(model)
        model_g.move_to(DOWN * 0.9)
        in_start, in_end = cards.get_bottom() + DOWN * 0.05, model.get_top() + LEFT * 0.6
        arrow_in = Arrow(in_start, in_end, color=GREY)
        self.play(FadeIn(model_g), GrowArrow(arrow_in))
        self.flow(Line(in_start, in_end), color=BLUE, n=7, run_time=1.2)

        mat = token_card("mat", GREEN).move_to(guess)
        out_start, out_end = model.get_top() + RIGHT * 0.6, guess.get_bottom()
        arrow_out = Arrow(out_start, out_end, color=GREEN)
        self.flow(Line(out_start, out_end), color=GREEN, n=5, run_time=0.9, sound=False)
        self.sfx("pop", gain=-4)
        self.say('The model guesses: "mat".', GrowArrow(arrow_out), ReplacementTransform(guess, mat),
                 Flash(guess.get_center(), color=GREEN, flash_radius=0.7))

        self.say(
            "The computer cuts text into small pieces called tokens.",
            *[Indicate(c, color=YELLOW) for c in [*cards, mat]],
            run_time=1.5,
        )
        self.say("Here, each token is one word.")

        new_row = VGroup(*cards, mat)
        next_guess = token_card("?", YELLOW, width=1.1)
        target = VGroup(new_row.copy(), next_guess).arrange(RIGHT, buff=0.2).move_to(UP * 1.6)
        self.say(
            "The new token joins the text.",
            FadeOut(arrow_out),
            FadeOut(arrow_in),
            Transform(new_row, target[0]),
            FadeIn(next_guess.move_to(target[1])),
        )
        arrow_in2 = Arrow(new_row.get_bottom(), model.get_top() + LEFT * 0.6, color=GREY)
        arrow_out2 = Arrow(model.get_top() + RIGHT * 0.6, next_guess.get_bottom(), color=GREEN)
        dot_card = token_card(".", GREEN, width=1.1).move_to(next_guess)
        self.sfx("pop", gain=-6, offset=0.5)
        self.say(
            "Then the model guesses again.",
            GrowArrow(arrow_in2),
            GrowArrow(arrow_out2),
            ReplacementTransform(next_guess, dot_card),
        )

        loop = Arc(radius=0.45, start_angle=PI / 2, angle=-1.7 * PI, color=YELLOW, stroke_width=5)
        loop.add_tip(tip_length=0.2)
        loop.next_to(model_g, RIGHT, buff=0.3)
        loop_lbl = T("repeat", 26, YELLOW).next_to(loop, RIGHT, buff=0.2)
        self.sfx("tick", gain=-6)
        self.say("Guess, add, guess again. One token at a time.", Create(loop), FadeIn(loop_lbl), run_time=1.4)
        self.say("Running a model like this is called inference.",
                 Circumscribe(VGroup(model_g, loop, loop_lbl), color=YELLOW))
        self.end()


class S02_Numbers(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S02_Numbers"])
        self.say("Computers only understand numbers.")

        words = [("The", [0.1, 0.7]), ("cat", [0.9, 0.3]), ("sat", [0.4, 0.8])]
        rows = VGroup()
        for w, v in words:
            card = token_card(w, width=1.4)
            arrow = Arrow(LEFT * 0.4, RIGHT * 0.4, color=GREY, buff=0)
            rows.add(VGroup(card, arrow, vec(v)).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to(LEFT * 3.4 + UP * 0.4)

        self.say("So each token becomes a list of numbers.",
                 LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.2), wait=0.2)
        self.sfx("pop", gain=-8)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(r[1]), FadeIn(r[2], shift=RIGHT * 0.2)) for r in rows],
                              lag_ratio=0.25), run_time=1.6)
        self.say("A list of numbers is called a vector.", *[Indicate(r[2], color=TEAL) for r in rows])
        note = T("(made-up numbers; real vectors have thousands)", 20, GREY).next_to(rows, DOWN, buff=0.35)
        self.play(FadeIn(note))

        axes = Axes(x_range=[0, 1, 0.25], y_range=[0, 1, 0.25], x_length=4.2, y_length=4.2,
                    axis_config={"color": DIM, "include_tip": False}).move_to(RIGHT * 3.6 + UP * 0.3)
        pts = {"cat": ([0.82, 0.30], BLUE), "kitten": ([0.90, 0.42], BLUE),
               "dog": ([0.70, 0.22], BLUE), "car": ([0.12, 0.88], ORANGE)}
        dots = VGroup()
        for name, (xy, color) in pts.items():
            d = Dot(axes.c2p(*xy), color=color, radius=0.09)
            ring = Dot(axes.c2p(*xy), color=color, radius=0.22, fill_opacity=0.18)
            lbl = T(name, 22, color).next_to(d, UP if name != "dog" else DOWN, buff=0.12)
            dots.add(VGroup(ring, d, lbl))
        self.say("Each vector is a point on a map.", Create(axes), run_time=1.2, wait=0.2)
        self.sfx("pop", gain=-8)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.2))
        cluster = VGroup(*dots[:3])
        self.say("Tokens with similar meaning sit close together.", Circumscribe(cluster, color=BLUE))
        self.say('"cat" and "kitten" are close. "car" is far away.', Indicate(dots[3], color=ORANGE))
        self.end()


class S03_Tower(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S03_Tower"])
        labels = ["layer 1", "layer 2", "layer 3", "...", "layer 46", "layer 47", "layer 48"]
        floors = VGroup()
        for i, lbl in enumerate(labels):
            if lbl == "...":
                floors.add(VGroup(T("...", 26, GREY)))
            else:
                floors.add(floor_3d(3.0, 0.48, BLUE, top=(i == len(labels) - 1), label=lbl))
        floors.arrange(UP, buff=0.07).move_to(UP * 0.15 + LEFT * 0.15)
        self.say("The model is like a tall tower.",
                 LaggedStart(*[FadeIn(f, shift=UP * 0.15) for f in floors], lag_ratio=0.12), run_time=1.6)
        self.say("Each floor is called a layer. This model has 48 layers.")

        inp = T("The cat sat on the", 26, GREY).next_to(floors, DOWN, buff=0.3)
        self.play(FadeIn(inp))
        runner = vec([0.4, 0.8], color=BLUE, size=20)
        runner.next_to(floors, LEFT, buff=0.5).align_to(floors[0], DOWN)
        runner_glow = halo(runner, BLUE, buff=0.06, opacity=0.06)
        self.play(FadeIn(runner_glow), FadeIn(runner, shift=UP * 0.2))
        self.say("The vectors go up the tower, floor by floor.", run_time=0.6, wait=0.2)
        colors = [BLUE, BLUE, TEAL, TEAL, GREEN, GREEN, GREEN]
        for floor, color in zip(floors, colors):
            self.sfx("tick", gain=-10)
            anims = [runner.animate.set_y(floor.get_y()).set_color(color),
                     runner_glow.animate.set_y(floor.get_y()).set_stroke(color)]
            if len(floor) > 1:
                anims.append(Indicate(floor[1], color=YELLOW, scale_factor=1.04))
            self.play(*anims, run_time=0.45)
        self.say("Each layer changes the vectors a little. They get smarter.")

        out = token_card("mat", GREEN).next_to(floors, UP, buff=0.3)
        self.sfx("pop", gain=-4)
        self.say('At the top, the model picks the next token: "mat".', FadeIn(out, shift=UP * 0.2),
                 FadeIn(glow(out[0], GREEN)), Flash(out.get_center(), color=GREEN, flash_radius=0.8))
        star = T("attention", 28, YELLOW, weight=BOLD, font=DISPLAY).next_to(floors, RIGHT, buff=0.8)
        arrows = VGroup(*[Arrow(star.get_left(), floors[i][1].get_right(), color=YELLOW, buff=0.1,
                                stroke_width=3) for i in (1, 4, 6)])
        self.say("Many layers have a special skill, called attention.", FadeIn(star),
                 *[GrowArrow(a) for a in arrows])
        self.end()


class S04_Attention(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S04_Attention"])
        names = ["The", "cat", "sat"]
        keys = [[1, 0], [0, 1], [1, 1]]
        values = [[1, 0], [0, 1], [2, 2]]
        ys = [1.25, 0.15, -0.95]

        cards = VGroup(*[token_card(n, width=1.3).move_to([-5.7, y, 0]) for n, y in zip(names, ys)])
        self.say("In attention, the newest token looks back at all tokens.",
                 LaggedStart(*[FadeIn(c) for c in cards], lag_ratio=0.2))
        self.say('Here the model works on the token "sat".', Indicate(cards[2], color=YELLOW))

        q_label = T("query of sat", 24, YELLOW).move_to([3.4, 2.55, 0])
        q_vec = vec([2, 0], color=YELLOW).next_to(q_label, RIGHT, buff=0.3)
        q_icon = T("?", 34, YELLOW, weight=BOLD).next_to(q_vec, RIGHT, buff=0.2)
        self.say("Every token makes three vectors.")
        self.sfx("pop", gain=-8)
        self.say('Query: a question. "Which tokens matter to me?"', FadeIn(q_label), FadeIn(q_vec),
                 FadeIn(q_icon), FadeIn(halo(q_vec, YELLOW, buff=0.05)))

        k_head = VGroup(T("key", 24, TEAL, weight=BOLD), T("(name tag)", 20, TEAL)).arrange(DOWN, buff=0.06)
        k_head.move_to([-3.35, 2.2, 0])
        v_head = VGroup(T("value", 24, ORANGE, weight=BOLD), T("(backpack)", 20, ORANGE)).arrange(DOWN, buff=0.06)
        v_head.move_to([-1.0, 2.2, 0])
        k_vecs = VGroup(*[vec(k, color=TEAL).move_to([-3.35, y, 0]) for k, y in zip(keys, ys)])
        v_vecs = VGroup(*[vec(v, color=ORANGE).move_to([-0.95, y, 0]) for v, y in zip(values, ys)])
        self.sfx("pop", gain=-10)
        self.say("Key: a name tag. It says what the token contains.", FadeIn(k_head),
                 LaggedStart(*[FadeIn(k, shift=RIGHT * 0.2) for k in k_vecs], lag_ratio=0.2))
        self.sfx("pop", gain=-10)
        self.say("Value: a backpack. It holds the token's information.", FadeIn(v_head),
                 LaggedStart(*[FadeIn(v, shift=RIGHT * 0.2) for v in v_vecs], lag_ratio=0.2))
        self.say("The model makes them with tables of numbers it learned.")

        # Step 1: scores.
        step = T("Step 1: compare with each name tag", 24, YELLOW).move_to([3.9, 2.0, 0])
        dots_txt = ["2×1 + 0×0 = 2", "2×0 + 0×1 = 0", "2×1 + 0×1 = 2"]
        prods = VGroup(*[M(s, 24).move_to([3.2, y, 0]) for s, y in zip(dots_txt, ys)])
        self.say("Step 1: compare the question with each name tag.", FadeIn(step), wait=0.3)
        for i in range(3):
            self.sfx("tick", gain=-8)
            self.play(Indicate(q_vec, color=YELLOW, scale_factor=1.05),
                      Indicate(k_vecs[i], color=TEAL, scale_factor=1.05),
                      FadeIn(prods[i], shift=LEFT * 0.2), run_time=0.9)
        self.say("Multiply in pairs. Then add. A big number means a good match.")

        scores = VGroup(*[M(s, 26, YELLOW).move_to([5.9, y, 0]) for s, y in zip(["1.414", "0", "1.414"], ys)])
        div = T("divide by √2 = 1.414", 22, GREY).move_to([5.6, -1.85, 0])
        self.say("Then divide by 1.414. That is the square root of 2.", FadeIn(div),
                 *[TransformFromCopy(p, s) for p, s in zip(prods, scores)])
        self.say("Why 2? Each vector here has 2 numbers. This keeps scores small.")
        self.play(FadeOut(prods), FadeOut(div), FadeOut(step))
        score_title = T("score", 22, YELLOW).next_to(scores[0], UP, buff=0.12)
        self.play(scores.animate.shift(LEFT * 4.2), FadeIn(score_title.shift(LEFT * 4.2 + UP * 0.05)))

        # Step 2: shares.
        step2 = T("Step 2: turn scores into shares", 24, YELLOW).move_to([3.9, 2.0, 0])
        exps = VGroup(*[M(s, 24).move_to([4.7, y, 0])
                        for s, y in zip(["e^1.414 = 4.113", "e^0 = 1", "e^1.414 = 4.113"], ys)])
        self.say("Step 2: turn scores into shares that add up to 1.", FadeIn(step2))
        self.say("Raise e to each score. e is a fixed number, about 2.718.",
                 *[TransformFromCopy(s, e) for s, e in zip(scores, exps)],
                 speak="Raise e to each score. e is a fixed number, about two point seven one eight.")
        total = M("total = 4.113 + 1 + 4.113 = 9.227", 22, GREY).move_to([3.4, -1.85, 0])
        self.say("Add them up. The total is 9.227.", FadeIn(total))

        shares = [0.446, 0.108, 0.446]
        bars = VGroup()
        for s, y in zip(shares, ys):
            bar = Rectangle(width=s * 5.0, height=0.42, stroke_width=0, fill_color=YELLOW, fill_opacity=0.85)
            bar.set_sheen(0.4, UP)
            bar.move_to([2.6, y, 0], aligned_edge=LEFT)
            lbl = M(f"{s:.3f}", 22).next_to(bar, RIGHT, buff=0.12)
            bars.add(VGroup(bar, lbl))
        self.sfx("swell", gain=-10)
        self.say("Divide each one by the total. Now they add up to 1.", FadeOut(exps),
                 *[GrowFromEdge(b[0], LEFT) for b in bars], *[FadeIn(b[1]) for b in bars])
        self.say('"The" and "sat" get big shares. "cat" gets a small share.',
                 Indicate(bars[0], color=YELLOW), Indicate(bars[2], color=YELLOW))
        self.play(FadeOut(total), FadeOut(step2))

        # Step 3: mix the values.
        step3 = T("Step 3: mix the backpacks", 24, YELLOW).move_to([3.9, 2.0, 0])
        mix_txt = ["0.446 × [1, 0] = [0.446, 0]", "0.108 × [0, 1] = [0, 0.108]",
                   "0.446 × [2, 2] = [0.892, 0.892]"]
        mixes = VGroup(*[M(s, 22).move_to([3.7, y, 0]) for s, y in zip(mix_txt, ys)])
        self.say("Step 3: mix the backpacks by the shares.", FadeIn(step3), FadeOut(bars), FadeOut(scores),
                 FadeOut(score_title), wait=0.3)
        for i in range(3):
            self.sfx("tick", gain=-8)
            self.play(Indicate(v_vecs[i], color=ORANGE, scale_factor=1.05), FadeIn(mixes[i], shift=LEFT * 0.2),
                      run_time=0.8)
        line = Line([1.5, -1.5, 0], [6.6, -1.5, 0], color=GREY)
        result = M("sum = [1.338, 1.000]", 26, GREEN).move_to([4.6, -2.0, 0])
        self.sfx("chime", gain=-6)
        self.say('Add them up. This is the new vector for "sat".', Create(line), FadeIn(result, shift=UP * 0.2),
                 FadeIn(halo(result, GREEN, buff=0.12)))
        self.say('"sat" now carries mostly "The" and "sat", and only a little of "cat".',
                 Circumscribe(result, color=GREEN))

        self.clean()
        f0 = M("output =", 38)
        f1 = M("softmax(", 38, YELLOW).next_to(f0, RIGHT, buff=0.3)
        f2 = M("Q · K^T / √d", 38, TEAL).next_to(f1, RIGHT, buff=0.1)
        f3 = M(")", 38, YELLOW).next_to(f2, RIGHT, buff=0.1)
        f4 = M("· V", 38, ORANGE).next_to(f3, RIGHT, buff=0.3)
        formula = VGroup(f0, f1, f2, f3, f4).move_to(UP * 1.0)
        frame = RoundedRectangle(corner_radius=0.2, width=formula.width + 1.0, height=formula.height + 0.7,
                                 stroke_color=YELLOW, stroke_width=2, fill_color="#0C121D", fill_opacity=0.85)
        frame.move_to(formula)
        self.sfx("swell", gain=-8)
        self.say("All three steps fit in one line.", FadeIn(glow(frame, YELLOW, opacity=0.035)), FadeIn(frame),
                 Write(formula), run_time=1.8)
        tags = VGroup(
            T("Step 1: scores", 24, TEAL).next_to(frame, DOWN, buff=0.3).set_x(f2.get_x()),
            T("Step 2: shares", 24, YELLOW).next_to(frame, DOWN, buff=0.85).align_to(f1, LEFT),
            T("Step 3: mix", 24, ORANGE).next_to(frame, DOWN, buff=0.85).align_to(f4, RIGHT),
        )
        self.say("This is the attention formula.",
                 LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in tags], lag_ratio=0.3))
        legend = T("Q: all queries    K: all keys    V: all values    d: numbers per vector", 22, GREY)
        legend.move_to(DOWN * 1.75)
        self.say("Big letters mean all tokens at once, as a table.", FadeIn(legend))
        self.end()


class S05_NoCache(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S05_NoCache"])
        words = ["The", "cat", "sat", "on", "the"]
        self.say("Now the model writes many tokens, one by one.")
        self.say("Each new token must look back at all tokens so far.")
        self.say("Without memory, it makes every key and value again.", wait=0.3)

        counter_lbl = T("jobs this step:", 24, GREY).move_to([3.3, -0.7, 0])
        counter = M("0", 30, YELLOW).next_to(counter_lbl, RIGHT, buff=0.25)
        total_lbl = T("total jobs:", 24, GREY).next_to(counter_lbl, DOWN, buff=0.35, aligned_edge=LEFT)
        total = M("0", 30, RED).next_to(total_lbl, RIGHT, buff=0.25)
        self.play(FadeIn(counter_lbl), FadeIn(counter), FadeIn(total_lbl), FadeIn(total))

        cards = VGroup()
        pairs = VGroup()
        running = 0
        stair = VGroup()
        for t in range(1, 6):
            card = token_card(words[t - 1], width=1.15, height=0.7, size=26)
            if t == 1:
                card.move_to([-5.2, 1.9, 0])
            else:
                card.next_to(cards[-1], RIGHT, buff=0.15)
            cards.add(card)
            self.sfx("pop", gain=-12)
            self.play(FadeIn(card, shift=DOWN * 0.2), run_time=0.4)
            new_pairs = VGroup(*[kv_pair(0.9).next_to(c, DOWN, buff=0.2) for c in cards])
            running += t
            color = interpolate_color(ManimColor(YELLOW), ManimColor(RED), (t - 1) / 4)
            col = VGroup(*[Square(0.32, stroke_color=color, fill_color=color, fill_opacity=0.55)
                           for _ in range(t)]).arrange(UP, buff=0.04)
            col.move_to([-5.2 + (t - 1) * 0.4, -2.35, 0], aligned_edge=DOWN)
            for k in range(t):
                self.sfx("tick", gain=-12, offset=0.08 * k)
            self.play(FadeOut(pairs), LaggedStart(*[FadeIn(p, scale=1.4) for p in new_pairs], lag_ratio=0.1),
                      Transform(counter, M(str(t), 30, YELLOW).move_to(counter)),
                      Transform(total, M(str(running), 30, RED).move_to(total, aligned_edge=LEFT)),
                      FadeIn(col, shift=UP * 0.2), run_time=0.9)
            pairs = new_pairs
            stair.add(col)
            if t == 3:
                self.say("Token 1: 1 job. Token 2: 2 jobs. Token 3: 3 jobs.", run_time=0.6, wait=0.3)
        self.say("The work piles up like stairs.", Indicate(stair, color=RED))
        self.clean()

        eq1 = M("1 + 2 + 3 + ... + n  =  n × (n + 1) ÷ 2", 32).move_to(UP * 1.0)
        eq2 = M("1,000 × 1,001 ÷ 2  =  500,500", 34, RED).next_to(eq1, DOWN, buff=0.6)
        self.say("For 1,000 tokens, add 1 + 2 + 3, all the way to 1,000.", Write(eq1), run_time=1.6,
                 speak="For one thousand tokens, add one plus two plus three, all the way to one thousand.")
        self.sfx("buzz", gain=-16)
        self.say("That is 500,500 jobs. Very tiring!", FadeIn(eq2, shift=UP * 0.2),
                 FadeIn(halo(eq2, RED, buff=0.15)))
        self.end()


class S06_KVCache(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S06_KVCache"])
        a = kv_pair(1.6).move_to([-2.5, 1.0, 0])
        b = kv_pair(1.6).move_to([2.5, 1.0, 0])
        la = T("key and value of The, at token 4", 22, GREY).next_to(a, DOWN, buff=0.25)
        lb = T("key and value of The, at token 5", 22, GREY).next_to(b, DOWN, buff=0.25)
        eq = T("=", 60, GREEN).move_to([0, 1.0, 0])
        self.say("But wait. Old keys and values never change!", FadeIn(a), FadeIn(b), FadeIn(la), FadeIn(lb),
                 FadeIn(eq))
        self.say("So why make them again?")
        self.clean()

        book = notebook(width=6.4, height=1.7).move_to([1.6, -0.75, 0])
        self.sfx("swell", gain=-8)
        self.say("The model writes them in a notebook, and reads them later.", FadeIn(book, shift=UP * 0.2))
        self.say("This notebook is the KV cache. K is for keys. V is for values.", Indicate(book[1], color=YELLOW))

        counter_lbl = T("jobs this step:", 22, GREY).move_to([-4.6, -0.3, 0])
        counter = M("0", 28, YELLOW).next_to(counter_lbl, RIGHT, buff=0.2)
        total_lbl = T("total jobs:", 22, GREY).next_to(counter_lbl, DOWN, buff=0.3, aligned_edge=LEFT)
        total = M("0", 28, GREEN).next_to(total_lbl, RIGHT, buff=0.2)
        self.play(FadeIn(counter_lbl), FadeIn(counter), FadeIn(total_lbl), FadeIn(total))

        words = ["The", "cat", "sat", "on", "the"]
        stored = VGroup()
        for t in range(1, 6):
            card = token_card(words[t - 1], width=1.15, height=0.7, size=26)
            card.move_to([-2.2 + (t - 1) * 1.3, 1.6, 0])
            pair = kv_pair(0.9).next_to(card, DOWN, buff=0.2)
            slot = [-0.55 + (t - 1) * 1.1, -0.75, 0]
            self.sfx("pop", gain=-10)
            self.play(FadeIn(card, shift=DOWN * 0.2), FadeIn(pair, scale=1.4), run_time=0.5)
            self.sfx("tick", gain=-10, offset=0.6)
            self.play(pair.animate.move_to(slot),
                      Transform(counter, M("1", 28, YELLOW).move_to(counter)),
                      Transform(total, M(str(t), 28, GREEN).move_to(total, aligned_edge=LEFT)),
                      *[Indicate(s, color=YELLOW, scale_factor=1.1) for s in stored], run_time=0.8)
            stored.add(pair)
            if t == 2:
                self.say("Each new token makes only its own key and value.", run_time=0.6, wait=0.3)
        self.say("The old ones are read from the KV cache.")
        self.clean()

        base = -2.2
        tall = Rectangle(width=1.6, height=4.2, stroke_width=0, fill_color=RED, fill_opacity=0.85)
        tall.set_sheen(0.35, LEFT)
        tall.move_to([-2.0, base, 0], aligned_edge=DOWN)
        tiny = Rectangle(width=1.6, height=0.04, stroke_width=0, fill_color=GREEN, fill_opacity=0.95)
        tiny.move_to([2.0, base, 0], aligned_edge=DOWN)
        floor_line = Line([-4.0, base, 0], [4.0, base, 0], color=DIM)
        lt = VGroup(M("500,500", 28, RED), T("without KV cache", 22, GREY)).arrange(DOWN, buff=0.1)
        lt.next_to(tall, UP, buff=0.15)
        ls = VGroup(M("1,000", 28, GREEN), T("with KV cache", 22, GREY)).arrange(DOWN, buff=0.1)
        ls.next_to(tiny, UP, buff=0.15)
        self.sfx("swell", gain=-8)
        self.say("For 1,000 tokens: 1,000 jobs instead of 500,500.", Create(floor_line),
                 GrowFromEdge(tall, DOWN), GrowFromEdge(tiny, DOWN), FadeIn(lt), FadeIn(ls), run_time=1.4)
        self.say("That is about 500 times less work, in each layer.", FadeIn(halo(ls, GREEN, buff=0.15)))
        self.say("The price: the KV cache uses memory.")
        self.end()


class S07_Blocks(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S07_Blocks"])
        chip_icon = gpu_chip().move_to([-3.6, 0.5, 0])
        self.sfx("pop", gain=-8)
        self.say("A GPU is the chip that does the math.", FadeIn(chip_icon, scale=0.85))
        vllm = VGroup(
            RoundedRectangle(corner_radius=0.15, width=3.4, height=1.0, stroke_color=BLUE, fill_color=BLUE,
                             fill_opacity=0.15),
            T("vLLM", 40, BLUE, weight=BOLD, font=DISPLAY),
        )
        vllm[1].move_to(vllm[0])
        vllm.move_to([2.6, 0.5, 0])
        link = Arrow(vllm.get_left(), chip_icon.get_right() + RIGHT * 0.15, color=BLUE)
        self.say("vLLM is the program that runs the model on GPUs.", FadeIn(vllm), GrowArrow(link))
        self.clean()
        self.say("vLLM keeps the KV cache in pages. A page is called a block.")

        def block_grid():
            cells = VGroup(*[Square(0.42, stroke_color=YELLOW, stroke_width=2, fill_opacity=0) for _ in range(16)])
            cells.arrange_in_grid(rows=4, cols=4, buff=0.04)
            frame = SurroundingRectangle(cells, color=YELLOW, buff=0.1, corner_radius=0.08)
            return VGroup(glow(frame, YELLOW, opacity=0.03), frame, cells)

        b1 = block_grid().move_to([-3.8, 0.55, 0])
        b2 = block_grid().move_to([-1.1, 0.55, 0])
        l1 = T("block 1", 22, YELLOW).next_to(b1, DOWN, buff=0.15)
        l2 = T("block 2", 22, YELLOW).next_to(b2, DOWN, buff=0.15)
        self.play(FadeIn(b1), FadeIn(l1))
        self.say("In the vLLM test, one block holds 16 tokens.", run_time=0.6, wait=0.3)
        fills = VGroup(*[c.copy().set_fill(TEAL, opacity=0.65).set_stroke(TEAL) for c in b1[2]])
        for k in range(16):
            self.sfx("tick", gain=-14, offset=k * 0.12)
        self.play(LaggedStart(*[FadeIn(f, scale=0.5) for f in fills], lag_ratio=0.08), run_time=2.0)
        self.sfx("pop", gain=-8)
        self.say("When a block is full, vLLM takes a new block.", FadeIn(b2), FadeIn(l2), wait=0.2)
        f2 = VGroup(*[b2[2][i].copy().set_fill(TEAL, opacity=0.65).set_stroke(TEAL) for i in range(5)])
        self.play(LaggedStart(*[FadeIn(f, scale=0.5) for f in f2], lag_ratio=0.15), run_time=1.0)

        q = T("How much memory is one block?", 28, YELLOW).move_to([3.6, 1.2, 0])
        self.say("How big is one block?", FadeIn(q))
        self.clean()

        terms = ["heads", "×", "tokens per block", "×", "(key length + value length)", "×", "bytes per number"]
        nums = ["2", "×", "16", "×", "(64 + 64)", "×", "4"]
        colors = [PURPLE, GREY, YELLOW, GREY, TEAL, GREY, ORANGE]
        head = T("bytes per block =", 30).move_to(UP * 1.9)
        trow = VGroup(*[T(t, 26, c) for t, c in zip(terms, colors)]).arrange(RIGHT, buff=0.2).move_to(UP * 0.9)
        nrow = VGroup(*[(T(n, 30, c) if n == "×" else M(n, 32, c)).move_to([t.get_x(), trow.get_y() - 0.75, 0])
                        for t, n, c in zip(trow, nums, colors)])
        self.say("vLLM uses this formula for one block of one layer.", FadeIn(head), FadeIn(trow), run_time=1.2)
        self.say("A head is one copy of the attention math. The test uses 2.", Indicate(trow[0], color=PURPLE))
        self.say("Now put in the numbers from the vLLM test.",
                 *[TransformFromCopy(t, n) for t, n in zip(trow, nrow)], run_time=1.4)
        res = M("= 16,384 bytes", 40, GREEN).next_to(nrow, DOWN, buff=0.55)
        self.sfx("chime", gain=-6)
        self.say("2 × 16 × 128 × 4 = 16,384 bytes.", FadeIn(res, shift=UP * 0.2), FadeIn(halo(res, GREEN, buff=0.15)))
        self.say("A byte is the smallest piece of computer memory.")
        self.end()


class S08_TwoKinds(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S08_TwoKinds"])
        self.say("The model in the issue mixes two kinds of layers.")

        left = panel(5.6, 3.6, TEAL, "attention layer").move_to([-3.3, 0.45, 0])
        right = panel(5.6, 3.6, PURPLE, "state layer").move_to([3.3, 0.45, 0])
        self.play(FadeIn(left), FadeIn(right))

        blocks = VGroup()
        for i in range(4):
            r = Rectangle(width=0.9, height=0.9, stroke_color=YELLOW, fill_color=YELLOW, fill_opacity=0.3)
            r.set_sheen(0.35, UP + LEFT)
            r.move_to([-5.0 + i * 1.1, 0.25, 0])
            blocks.add(r)
        self.say("An attention layer keeps a key and a value for every token.")
        for k in range(4):
            self.sfx("pop", gain=-12, offset=0.8 + k * 0.5)
        self.say("So its KV cache grows with every token.",
                 LaggedStart(*[GrowFromEdge(b, LEFT) for b in blocks], lag_ratio=0.4), run_time=2.2)

        card = RoundedRectangle(corner_radius=0.12, width=2.2, height=1.2, stroke_color=PURPLE,
                                fill_color=PURPLE, fill_opacity=0.25)
        card.set_sheen(0.35, UP + LEFT)
        card.move_to([3.3, 0.25, 0])
        word = T("summary", 26).move_to(card)
        self.play(FadeIn(glow(card, PURPLE, opacity=0.04)), FadeIn(card), FadeIn(word))
        self.say("A state layer keeps one small summary instead.", wait=0.3)
        for w in ["summary v2", "summary v3", "summary v4"]:
            self.sfx("tick", gain=-8)
            self.play(Transform(word, T(w, 26).move_to(card)), Indicate(card, color=PURPLE, scale_factor=1.05),
                      run_time=0.6)
        self.say("It rewrites the summary for each token. Its memory stays the same size.")
        note = T("(in the real model, the state layers are GDN layers)", 20, GREY).next_to(right, DOWN, buff=0.2)
        self.play(FadeIn(note))
        self.clean()

        shelf_s = panel(5.8, 1.5, PURPLE, "state group").move_to([-3.2, 0.8, 0])
        shelf_a = panel(5.8, 1.5, TEAL, "attention group").move_to([3.2, 0.8, 0])
        self.say("vLLM puts layers into groups.", FadeIn(shelf_s), FadeIn(shelf_a), wait=0.2)
        s_chips = VGroup(*[chip(f"layer {i}", PURPLE) for i in (1, 2, 3)]).arrange(RIGHT, buff=0.15)
        s_chips.move_to(shelf_s[0].get_center() + DOWN * 0.2)
        a_chips = VGroup(*[chip(f"layer {i}", TEAL) for i in (4, 8, 12)]).arrange(RIGHT, buff=0.15)
        a_chips.move_to(shelf_a[0].get_center() + DOWN * 0.2)
        self.sfx("pop", gain=-10)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in [*s_chips, *a_chips]], lag_ratio=0.1))
        self.say("The layers in one group share one row of blocks.")
        self.say("Layers of different kinds never share a group.")
        note2 = T("(layer numbers chosen for illustration)", 20, GREY).move_to(DOWN * 1.0)
        self.play(FadeIn(note2))
        self.wait(0.8)
        self.end()


class S09_Split(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S09_Split"])
        self.say("This model is too big for one GPU.", wait=0.2)

        kinds = ["state" if (i % 4) != 3 else "attn" for i in range(48)]
        floor_h, gap = 0.07, 0.012
        tower = VGroup(*[
            Rectangle(width=1.6, height=floor_h, stroke_width=0,
                      fill_color=PURPLE if k == "state" else TEAL, fill_opacity=0.9)
            for k in kinds
        ]).arrange(UP, buff=gap).move_to(UP * 0.2)
        tower_glow = halo(tower, BLUE, buff=0.05, opacity=0.03)
        lbl = T("48 layers", 24).next_to(tower, UP, buff=0.15)
        legend = VGroup(
            VGroup(Square(0.22, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0),
                   T("state layer", 20)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.22, fill_color=TEAL, fill_opacity=0.85, stroke_width=0),
                   T("attention layer", 20)).arrange(RIGHT, buff=0.12),
            T("(pattern for illustration)", 18, GREY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.6, 1.8, 0])
        self.play(FadeIn(tower_glow), LaggedStart(*[FadeIn(f) for f in tower], lag_ratio=0.02), FadeIn(lbl),
                  FadeIn(legend), run_time=1.5)

        sizes = [2, 16, 16, 14]
        xs = [-4.5, -1.5, 1.5, 4.5]
        parts, start = [], 0
        for s in sizes:
            parts.append(VGroup(*tower[start: start + s]))
            start += s
        base_y = -2.1
        moves, labels = [], VGroup()
        for i, (p, x) in enumerate(zip(parts, xs)):
            target = p.copy().arrange(UP, buff=gap).move_to([x, 0, 0]).align_to([0, base_y, 0], DOWN)
            moves.append(Transform(p, target))
            lab = VGroup(T(f"GPU {i + 1}", 24, YELLOW, weight=BOLD, font=DISPLAY),
                         T(f"{sizes[i]} layers", 20, GREY)).arrange(DOWN, buff=0.06)
            lab.next_to(target, UP, buff=0.2)
            labels.add(lab)
        self.sfx("whoosh", gain=-10)
        self.say("So we cut the tower into 4 parts. Each part runs on its own GPU.", FadeOut(lbl), FadeOut(legend),
                 FadeOut(tower_glow), *moves, run_time=1.6, wait=0.2)
        self.sfx("pop", gain=-10)
        self.play(LaggedStart(*[FadeIn(lab, shift=DOWN * 0.15) for lab in labels], lag_ratio=0.2))
        self.say("Each part is called a stage.")
        self.say("The user chose 2, 16, 16 and 14 layers. That makes 48.")
        self.say("Stage 1 has only 2 layers. Both are state layers.",
                 Circumscribe(VGroup(parts[0], labels[0]), color=YELLOW))
        self.clean()

        gpu = panel(9.0, 3.6, YELLOW, "GPU 1  (stage 1: layers 1-2)").move_to(UP * 0.4)
        shelf_s = panel(3.9, 1.6, PURPLE, "state group").move_to([-2.2, 0.0, 0])
        shelf_a = panel(3.9, 1.6, TEAL, "attention group").move_to([2.2, 0.0, 0])
        s_chips = VGroup(chip("layer 1", PURPLE), chip("layer 2", PURPLE)).arrange(RIGHT, buff=0.15)
        s_chips.move_to(shelf_s[0].get_center() + DOWN * 0.2)
        empty = T("empty", 26, GREY).move_to(shelf_a[0].get_center() + DOWN * 0.2)
        self.play(FadeIn(glow(gpu[0], YELLOW, opacity=0.03)), FadeIn(gpu), FadeIn(shelf_s), FadeIn(shelf_a))
        self.play(FadeIn(s_chips))
        self.say("So on GPU 1, the attention group is empty.", FadeIn(empty), Indicate(shelf_a, color=TEAL))
        self.say("Empty is fine. But the old code did not expect it.")
        self.end()


def group_card(name, color, names, sizes, x, y, width=5.6):
    box = panel(width, 2.3, color, name, title_size=24).move_to([x, y, 0])
    c = box[0].get_center()
    nl_head = T("name list", 20, GREY).move_to(c + LEFT * 1.4 + UP * 0.35)
    st_head = T("size table", 20, GREY).move_to(c + RIGHT * 1.2 + UP * 0.35)
    nl = VGroup(*[T(n, 22) for n in names]).arrange(DOWN, buff=0.12).next_to(nl_head, DOWN, buff=0.2)
    st = VGroup(*[M(f"{n}: {s}", 20) for n, s in zip(names, sizes)]).arrange(DOWN, buff=0.12)
    st.next_to(st_head, DOWN, buff=0.2)
    return VGroup(box, nl_head, st_head, nl, st)


class S10_Bug(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S10_Bug"])
        clip = panel(3.3, 3.0, GREEN, "plan").move_to([5.2, 0.3, 0])
        self.say("Each GPU writes a plan: the KV cache memory to reserve.", FadeIn(clip))
        self.say("Let us use the tiny copy of the bug from the vLLM test.")

        ga = group_card("group A", TEAL, ["layer 1", "layer 2"], ["16,384", "16,384"], -2.4, 1.15)
        gb = group_card("group B", BLUE, ["layer 3"], ["32,768"], -2.4, -1.35)
        self.say("Group A has layer 1 and layer 2. Group B has layer 3.", FadeIn(ga[0]), FadeIn(gb[0]),
                 FadeIn(ga[3]), FadeIn(gb[3]))
        own = VGroup(T("this GPU owns:", 22, YELLOW), T("layer 2", 26, YELLOW, weight=BOLD)).arrange(DOWN, buff=0.06)
        own.move_to([5.2, 2.45, 0])
        self.say("This GPU owns only layer 2.", FadeIn(own))
        self.say("In the old code, each group has two parts.", FadeIn(ga[1]), FadeIn(gb[1]), FadeIn(ga[2]),
                 FadeIn(gb[2]), FadeIn(ga[4]), FadeIn(gb[4]))
        self.say("The name list says which layers. The size table says how much memory.")

        self.say("Step 1, trim: keep only this GPU's layers.", wait=0.2)
        cut_a = VGroup(Line(ga[3][0].get_left(), ga[3][0].get_right(), color=RED, stroke_width=5),
                       Line(ga[4][0].get_left(), ga[4][0].get_right(), color=RED, stroke_width=5))
        self.sfx("tick", gain=-6)
        self.play(Create(cut_a), run_time=0.8)
        self.play(FadeOut(ga[3][0]), FadeOut(ga[4][0]), FadeOut(cut_a), run_time=0.6)
        cut_b = Line(gb[3][0].get_left(), gb[3][0].get_right(), color=RED, stroke_width=5)
        self.sfx("tick", gain=-6)
        self.say("Group B has no layers here. Its name list becomes empty.", Create(cut_b), wait=0.3)
        self.play(FadeOut(gb[3][0]), FadeOut(cut_b), run_time=0.5)
        empty_b = T("(empty)", 22, GREY).move_to(gb[3][0])
        self.play(FadeIn(empty_b))
        warn = SurroundingRectangle(gb[4], color=RED, buff=0.12)
        warn_lbl = T("not cut!", 22, RED, weight=BOLD).next_to(warn, RIGHT, buff=0.15)
        self.sfx("pop", gain=-6)
        self.say("But the old code forgot to cut its size table.", Create(warn), FadeIn(warn_lbl),
                 FadeIn(glow(warn, RED, opacity=0.05)))

        p2 = M("layer 2: 16,384", 19).move_to(clip[0].get_center() + UP * 0.2)
        p3 = M("layer 3: 32,768", 19, RED).next_to(p2, DOWN, buff=0.25)
        self.say("Step 2, plan: the old code read the size table.", TransformFromCopy(ga[4][1], p2),
                 TransformFromCopy(gb[4][0], p3), run_time=1.2)
        self.say("So the plan lists layer 3, a layer that lives on another GPU.", Indicate(p3, color=RED))

        self.say("Step 3, reserve: for each layer in the plan, find its group.", wait=0.2)
        a1 = Arrow(p2.get_left(), ga[3][1].get_right(), color=GREEN, buff=0.15)
        ok = check_mark().next_to(ga[3][1], LEFT, buff=0.2)
        self.sfx("chime", gain=-12)
        self.play(GrowArrow(a1), Create(ok), run_time=0.8)
        self.say("Layer 2 is in group A. Good.")
        self.play(FadeOut(a1))
        a2 = Arrow(p3.get_left(), ga[3][1].get_right() + DOWN * 0.3, color=RED, buff=0.15)
        no_a = x_mark().next_to(ga[0], LEFT, buff=0.1).shift(DOWN * 0.2)
        self.sfx("tick", gain=-6)
        self.play(GrowArrow(a2), Create(no_a), run_time=0.7)
        self.play(FadeOut(a2))
        a3 = Arrow(p3.get_left(), empty_b.get_right(), color=RED, buff=0.15)
        no_b = x_mark().next_to(gb[0], LEFT, buff=0.1)
        self.sfx("tick", gain=-6)
        self.play(GrowArrow(a3), Create(no_b), run_time=0.7)
        self.say("Layer 3 is in no name list. The search finds nothing.")

        self.clean()
        boom = VGroup(M("StopIteration", 56, RED), T("an error with no message", 30, GREY))
        boom.arrange(DOWN, buff=0.35).move_to(UP * 0.6)
        flash_box = SurroundingRectangle(boom, color=RED, buff=0.4, corner_radius=0.2)
        self.sfx("buzz", gain=-4)
        self.say("vLLM stops with an error that has no message.", FadeIn(boom, scale=1.2), Create(flash_box),
                 FadeIn(glow(flash_box, RED, opacity=0.06)), wait=0.1)
        self.play(Flash(flash_box.get_center(), color=RED, line_length=0.6, num_lines=16, flash_radius=2.6),
                  Wiggle(VGroup(boom, flash_box), scale_value=1.04, rotation_angle=0.02))
        self.say("On the real model, GPU 1 planned memory for layers on GPUs 2, 3 and 4.")
        self.end()


class S11_Fix(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S11_Fix"])
        ga = group_card("group A", TEAL, ["layer 2"], ["16,384"], -2.4, 1.15)
        gb = group_card("group B", BLUE, ["layer 3"], ["32,768"], -2.4, -1.35)
        gb[3][0].become(T("(empty)", 22, GREY).move_to(gb[3][0]))
        clip = panel(3.3, 3.0, GREEN, "plan").move_to([5.2, 0.3, 0])
        p2 = M("layer 2: 16,384", 19).move_to(clip[0].get_center() + UP * 0.2)
        p3 = M("layer 3: 32,768", 19, RED).next_to(p2, DOWN, buff=0.25)
        self.play(FadeIn(ga), FadeIn(gb), FadeIn(clip), FadeIn(p2), FadeIn(p3))

        fix1 = T("change 60234 · 27 Oct 2026", 22, GREEN).move_to([2.6, 2.75, 0])
        self.say("Fix 1: the plan step reads the name list instead.", FadeIn(fix1), wait=0.2)
        hl = SurroundingRectangle(VGroup(ga[3], gb[3]), color=GREEN, buff=0.1)
        self.play(Create(hl))
        self.say("Group B's name list is empty, so the plan skips group B.", FadeOut(p3, shift=RIGHT * 0.3))
        ok = check_mark().next_to(clip, DOWN, buff=0.15)
        clip_glow = glow(clip[0], GREEN, opacity=0.045)
        self.sfx("chime", gain=-4)
        self.say("Now the plan has only layer 2. No error.", Create(ok), FadeIn(clip_glow))
        self.play(FadeOut(hl))

        fix2 = T("change 61452 · 12 Nov 2026", 22, GREEN).move_to([2.6, 2.75, 0])
        self.say("Fix 2: each group now keeps only the size table.", Transform(fix1, fix2), wait=0.2)
        for g in (ga, gb):
            self.play(FadeOut(g[1]), FadeOut(g[3]), g[4].animate.move_to(g[0][0].get_center() + DOWN * 0.15),
                      g[2].animate.move_to(g[0][0].get_center() + UP * 0.45), run_time=0.8)
        self.play(Transform(gb[4], M("(empty)", 20, GREY).move_to(gb[4])))
        rule = T("names come from the size table", 24, YELLOW).next_to(clip, DOWN, buff=0.8).shift(LEFT * 0.8)
        self.say("The names come from that table. They can never disagree.", FadeIn(rule))
        self.say("A test in vLLM now checks this exact case.")
        self.sfx("chime", gain=-6)
        self.say("So issue 54711 is already fixed in the newest code.", Circumscribe(clip, color=GREEN),
                 speak="So issue five four seven one one is already fixed in the newest code.")
        self.end()


class S12_Summary(Explainer):
    def construct(self):
        self.chapter(CHAPTERS["S12_Summary"])
        names = ["text", "tokens", "vectors", "48 layers", "next token"]
        colors = [GREY, BLUE, TEAL, PURPLE, GREEN]
        boxes = VGroup(*[chip(n, c, size=24, height=0.7, pad=0.5) for n, c in zip(names, colors)])
        boxes.arrange(RIGHT, buff=0.6).move_to(UP * 1.8)
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.08, color=GREY)
                          for i in range(4)])
        back = CurvedArrow(boxes[4].get_top(), boxes[0].get_top(), angle=PI / 3.6, color=YELLOW)
        self.say("Inference: the model guesses one token at a time.",
                 LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15), *[GrowArrow(a) for a in arrows],
                 run_time=1.6, wait=0.1)
        self.play(Create(back))
        self.flow(*[Line(boxes[i].get_right(), boxes[i + 1].get_left()) for i in range(4)], color=BLUE, n=4,
                  run_time=1.2)
        self.flow(ArcBetweenPoints(boxes[4].get_top(), boxes[0].get_top(), angle=PI / 3.6), color=YELLOW, n=6,
                  run_time=1.2, sound=False)

        cache = chip("KV cache", YELLOW, size=24, height=0.7, pad=0.5).move_to([boxes[3].get_x(), -0.75, 0])
        link = Arrow(cache.get_top(), boxes[3].get_bottom(), buff=0.08, color=YELLOW)
        self.say("Attention layers read old keys and values from the KV cache.", FadeIn(cache),
                 FadeIn(glow(cache[0], YELLOW, opacity=0.05)), GrowArrow(link))
        chain = VGroup(chip("blocks", YELLOW, size=22), chip("groups", TEAL, size=22),
                       chip("stages (one per GPU)", ORANGE, size=22), chip("plan", GREEN, size=22))
        chain.arrange(RIGHT, buff=0.45).move_to(DOWN * 1.9)
        chain_arrows = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.06, color=GREY)
                                for i in range(3)])
        self.say("vLLM stores it in blocks, groups and stages, and plans the memory.", FadeIn(chain),
                 *[GrowArrow(a) for a in chain_arrows], run_time=1.4)
        self.say("In the bug, an empty group kept an uncut size table.", Indicate(chain[1], color=RED))
        self.say("The plan then named layers on other GPUs. The newest code fixes it.",
                 Indicate(chain[3], color=GREEN))
        self.clean()
        nxt = T("Next: issue 67028", 48, YELLOW, weight=BOLD, font=DISPLAY)
        self.wait_for_voice()
        self.sfx("chime", gain=-4)
        self.voice("Next: issue 67028.", speak="Next: issue six seven zero two eight.")
        self.play(FadeOut(self._caption), FadeIn(nxt, scale=1.1), FadeIn(halo(nxt, YELLOW, buff=0.3, opacity=0.03)))
        self._caption = None
        self.wait(1.5)
        self.end()
