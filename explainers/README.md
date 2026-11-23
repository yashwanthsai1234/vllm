# Explainers (local only, never pushed)

Our own folder of visual explainers for the vLLM issues we study. Each one has a
3Blue1Brown-style Manim video plus an interactive HTML page.

Keep this folder out of git. Run once from the repo root:

```bash
echo "explainers/" >> .git/info/exclude
echo "ISSUE_TRIAGE.md" >> .git/info/exclude
```

## Layout

```
explainers/
  .venv/                 Manim environment (separate from vLLM's own .venv)
  common/style.py        look: blueprint backdrop, glow, light particles, chapter cards,
                         narrated captions (say), sound cues (sfx), data flow (flow)
  common/audio.py        narration (macOS say), synthesized sound effects and music
  common/render.py       render all scenes -> one mp4 with ducked music + chapter times
  .cache/                generated voice lines, sound effects, music, backdrop (safe to delete)
  01-kv-cache-issue-54711/
    scenes.py            the video, one Manim scene per chapter (SCENES + CHAPTERS lists)
    index.html           video player with chapters + interactive panels
    try_attend.js        exercise: write attention yourself
    video/               rendered output (mp4, chapters.js, sheets/)
    media/               Manim cache (safe to delete)
```

## Setup (once)

```bash
cd explainers
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "manim>=0.19"
# pycairo needs: brew install cairo pkg-config
```

## Sound

- Every `say("...")` caption is also spoken. Narration overlaps the animations; the next
  caption waits for the voice. Fix how a word is spoken with `say(text, speak="...")` or add a
  rule to `SPEECH_FIX` in `common/audio.py` (numbers like 54711 should be spelled out).
- `self.sfx("pop" | "tick" | "whoosh" | "chime" | "buzz" | "swell", gain=-8)` adds a cue.
- Music is a calm synthesized piano-and-pad loop, mixed by `render.py` and ducked under the
  voice. `--no-music` turns it off; `--music-db -24` makes it quieter.
- Voice: Kokoro, an open neural TTS model that runs locally (`kokoro-onnx`, no PyTorch).
  One-time setup: `uv pip install --python .venv/bin/python kokoro-onnx soundfile`, and put
  `kokoro-v1.0.onnx` + `voices-v1.0.bin` from
  https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0 in `.cache/kokoro/`.
- Pick a voice: `EXPLAINER_VOICE=am_michael` (default `af_heart`; `af_*`/`am_*` are US,
  `bf_*`/`bm_*` are UK). Pace: `EXPLAINER_SPEED=0.95`. Old macOS voice: `EXPLAINER_ENGINE=say`.
- Check a pronunciation without rendering: `kokoro.tokenizer.phonemize(text, "en-us")`.

There is no LaTeX on this machine, so scenes use `T()` and `M()` text only
(no `MathTex`, `Tex`, `DecimalNumber`, `Integer`, `Brace.get_text`, or chart labels).

## Make or change a video

```bash
cd explainers
# fast draft + contact sheets (one PNG grid per scene) to check layout
.venv/bin/python common/render.py 01-kv-cache-issue-54711 --quality l --only S04_Attention --sheets
# final 1080p30, all scenes joined, chapters written
.venv/bin/python common/render.py 01-kv-cache-issue-54711 --quality h --fps 30
open 01-kv-cache-issue-54711/index.html
```

## Recipe for a new explainer

1. Read the issue, the linked PRs, and the current code first. Check every number
   against the code or a test. Note the checked facts at the top of `scenes.py`.
2. Copy `01-kv-cache-issue-54711/` to `NN-<topic>-issue-<number>/`.
3. Write the story as chapters. One running example from start to end, real numbers.
   Words: define each term once, then never swap in a synonym. Short sentences.
4. One scene class per chapter; list them in `SCENES` and `CHAPTERS`.
5. Draft-render with `--sheets`, look at every sheet, fix overlaps, then render final.
6. Adapt `index.html`: keep the player + chapters block, replace the panels.
