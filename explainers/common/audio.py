"""Narration, sound effects and music for explainer videos. All generated locally.

- Narration: Kokoro, an open neural TTS model (kokoro-onnx, runs locally on the CPU).
  Pick the voice with EXPLAINER_VOICE (default "af_heart"; also af_bella, am_michael,
  bf_emma, ...) and the pace with EXPLAINER_SPEED (default 1.0).
  Model files live in .cache/kokoro (kokoro-v1.0.onnx, voices-v1.0.bin) from
  https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
  EXPLAINER_ENGINE=say falls back to the macOS voice (EXPLAINER_VOICE="Samantha").
- Sound effects and music: synthesized with numpy, so there is no licensing question.
Everything is cached in explainers/.cache, keyed by content.
"""

import hashlib
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 44100
CACHE = Path(__file__).resolve().parents[1] / ".cache"
ENGINE = os.environ.get("EXPLAINER_ENGINE", "kokoro")
VOICE = os.environ.get("EXPLAINER_VOICE", "af_heart" if ENGINE == "kokoro" else "Samantha")
SPEED = float(os.environ.get("EXPLAINER_SPEED", "1.0"))
RATE = int(os.environ.get("EXPLAINER_RATE", "172"))  # words per minute, `say` only
_kokoro = None


def _kokoro_wav(text: str, path: Path):
    global _kokoro
    import soundfile as sf

    if _kokoro is None:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(str(CACHE / "kokoro" / "kokoro-v1.0.onnx"), str(CACHE / "kokoro" / "voices-v1.0.bin"))
    lang = "en-gb" if VOICE.startswith("b") else "en-us"
    samples, sr = _kokoro.create(text, voice=VOICE, speed=SPEED, lang=lang)
    sf.write(str(path), samples, sr)

# How to say words that TTS gets wrong. Applied to the caption before speaking.
SPEECH_FIX = [
    (r"\bvLLM\b", "V L L M"),
    (r"\bKV\b", "K V"),
    (r"\bGPUs\b", "G P U's"),
    (r"\bGPU\b", "G P U"),
    (r"\bGDN\b", "G D N"),
    (r"StopIteration", "Stop Iteration"),
    (r"×", " times "),
    (r"÷", " divided by "),
    (r"√", " square root of "),
    (r"\.\.\.", ", and so on, "),
    (r" = ", " equals "),
]


def speakable(text: str) -> str:
    for pat, rep in SPEECH_FIX:
        text = re.sub(pat, rep, text)
    return text


def _write_wav(path: Path, x: np.ndarray):
    x = np.clip(x, -1.0, 1.0)
    data = (x * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def narrate(text: str, speak: str | None = None) -> tuple[Path, float]:
    """TTS for one caption. Returns (wav path, seconds)."""
    spoken = speakable(speak or text)
    key = hashlib.sha1(f"{ENGINE}|{VOICE}|{SPEED}|{RATE}|{spoken}".encode()).hexdigest()[:16]
    out = CACHE / "tts" / f"{key}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = out.with_suffix(".raw.wav" if ENGINE == "kokoro" else ".aiff")
        if ENGINE == "kokoro":
            _kokoro_wav(spoken, raw)
        else:
            subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(raw), spoken], check=True)
        # Trim leading silence, pad 0.1 s, normalize the voice level.
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af",
             "silenceremove=start_periods=1:start_threshold=-50dB,adelay=100,loudnorm=I=-18:TP=-2",
             "-ar", str(SR), "-ac", "1", str(out)],
            check=True,
        )
        raw.unlink()
    return out, wav_duration(out)


def silence(seconds=0.1) -> Path:
    out = CACHE / "sfx" / f"silence_{seconds}.wav"
    if not out.exists():
        _write_wav(out, np.zeros(int(SR * seconds)))
    return out


# ---------------------------------------------------------------- sound effects

def _env(n, attack, decay_tau):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / decay_tau)


def _onepole_lowpass(x, cutoff_hz):
    """Time-varying one-pole low-pass. cutoff_hz can be a scalar or an array."""
    cut = np.broadcast_to(np.asarray(cutoff_hz, dtype=float), x.shape)
    alpha = 1 - np.exp(-2 * np.pi * cut / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += alpha[i] * (x[i] - acc)
        y[i] = acc
    return y


def _make_sfx(name: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    if name == "pop":
        n = int(SR * 0.16)
        t = np.arange(n) / SR
        f = 900 * np.exp(-t * 18) + 380
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.003, 0.045)
        return 0.55 * x
    if name == "tick":
        n = int(SR * 0.05)
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 2100 * t) * _env(n, 0.001, 0.008)
        return 0.35 * x
    if name == "whoosh":
        n = int(SR * 0.9)
        t = np.arange(n) / SR
        noise = rng.standard_normal(n)
        shape = np.sin(np.pi * np.clip(t / 0.9, 0, 1)) ** 2
        cutoff = 300 + 3500 * shape
        x = _onepole_lowpass(noise, cutoff) * shape
        return 0.5 * x / (np.abs(x).max() + 1e-9)
    if name == "chime":
        n = int(SR * 1.6)
        t = np.arange(n) / SR
        x = np.zeros(n)
        for start, f in [(0.0, 1318.5), (0.11, 1975.5)]:  # E6, B6
            s = int(start * SR)
            tt = t[: n - s]
            bell = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.76 * tt)
            x[s:] += bell * _env(n - s, 0.002, 0.45)
        return 0.28 * x / np.abs(x).max()
    if name == "buzz":
        n = int(SR * 0.75)
        t = np.arange(n) / SR
        x = np.zeros(n)
        for start, f in [(0.0, 98.0), (0.3, 82.4)]:
            s = int(start * SR)
            tt = t[: n - s]
            saw = 2 * ((f * tt) % 1) - 1
            x[s:] += saw * _env(n - s, 0.01, 0.16)
        x = _onepole_lowpass(x, 900)
        return 0.6 * x / np.abs(x).max()
    if name == "swell":
        n = int(SR * 1.2)
        t = np.arange(n) / SR
        x = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.3))
        shape = np.sin(np.pi * t / 1.2) ** 2
        return 0.16 * x * shape / 4
    raise KeyError(name)


def sfx(name: str) -> Path:
    out = CACHE / "sfx" / f"{name}.wav"
    if not out.exists():
        _write_wav(out, _make_sfx(name))
    return out


# ---------------------------------------------------------------- music

NOTE = {n: 440 * 2 ** ((i - 9) / 12) for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def _freq(name: str) -> float:
    return NOTE[name[:-1]] * 2 ** (int(name[-1]) - 4)


# A calm loop: Cmaj9 - Am7 - Fmaj7 - G6, then a variation.
PROGRESSION = [
    ["C3", "G3", "B3", "D4", "E4"],
    ["A2", "E3", "G3", "C4", "E4"],
    ["F2", "C3", "E3", "A3", "C4"],
    ["G2", "D3", "E3", "B3", "D4"],
    ["C3", "G3", "B3", "E4", "G4"],
    ["E2", "B2", "G3", "B3", "E4"],
    ["F2", "C3", "A3", "C4", "E4"],
    ["G2", "D3", "F3", "B3", "D4"],
]
CHORD_SECONDS = 4.8


def _piano(f, n, tau=1.4):
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in [(1, 1), (2, 0.45), (3, 0.18), (4, 0.08)])
    return x * _env(n, 0.004, tau)


def _music_loop() -> np.ndarray:
    seg = int(SR * CHORD_SECONDS)
    total = seg * len(PROGRESSION)
    x = np.zeros(total + SR * 6)
    t_seg = np.arange(seg + SR * 2) / SR
    for ci, chord in enumerate(PROGRESSION):
        s = ci * seg
        # Pad: soft sines, slow swell, overlapping the next chord.
        pad = sum(np.sin(2 * np.pi * _freq(nm) * t_seg) * (0.5 if i == 0 else 0.3)
                  for i, nm in enumerate(chord))
        shape = np.minimum(1, t_seg / 1.6) * np.minimum(1, np.maximum(0, (len(t_seg) / SR - t_seg) / 1.8))
        x[s : s + len(t_seg)] += 0.07 * pad * shape
        # Piano: a slow arpeggio, 8 notes per chord.
        pattern = [1, 2, 3, 4, 3, 2, 4, 3] if ci % 2 == 0 else [2, 3, 4, 3, 1, 3, 4, 2]
        step = seg // 8
        for k, idx in enumerate(pattern):
            f = _freq(chord[idx]) * (2 if idx >= 3 else 1)
            n = SR * 3
            p0 = s + k * step
            x[p0 : p0 + n] += 0.09 * _piano(f, n) * (1.0 if k % 4 == 0 else 0.7)
    # Wrap the tail into the start so the loop is seamless.
    loop = x[:total].copy()
    loop[: len(x) - total] += x[total:]
    # Simple reverb: a few feedback delays, applied circularly.
    wet = np.zeros_like(loop)
    for d, g in [(0.113, 0.32), (0.197, 0.26), (0.293, 0.2), (0.411, 0.15)]:
        wet += g * np.roll(loop, int(d * SR))
    y = loop + 0.6 * wet
    return 0.8 * y / np.abs(y).max()


def music(seconds: float) -> Path:
    """A music bed at least `seconds` long, built by repeating the loop."""
    loop_path = CACHE / "music" / "loop_v1.wav"
    if not loop_path.exists():
        _write_wav(loop_path, _music_loop())
    out = CACHE / "music" / f"bed_{int(seconds) + 1}.wav"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(loop_path),
             "-t", str(int(seconds) + 1), "-af", f"afade=t=in:d=3,afade=t=out:st={max(0, seconds - 4):.1f}:d=4",
             str(out)],
            check=True,
        )
    return out
