"""Render every scene of one explainer, join them, and write chapter times.

Usage (from the explainers folder):
    .venv/bin/python common/render.py 01-kv-cache-issue-54711 [--quality l|m|h] [--only S04_Attention]
    .venv/bin/python common/render.py 01-kv-cache-issue-54711 --sheets   # QA contact sheets

Output, inside the explainer folder:
    video/<folder>.mp4      the joined video
    video/chapters.json     [{"scene", "title", "start"}]
    video/chapters.js       the same list as `window.CHAPTERS`, loaded by index.html
    video/sheets/*.png      with --sheets: a grid of frames per scene, for review
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = {"l": ("-ql", "480p15"), "m": ("-qm", "720p30"), "h": ("-qh", "1080p60")}


def load_module(folder: Path):
    spec = importlib.util.spec_from_file_location("scenes", folder / "scenes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quality", default="h", choices=QUALITY)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--music-db", type=float, default=-21.0, help="music level under the voice")
    args = ap.parse_args()

    folder = (ROOT / args.folder).resolve()
    mod = load_module(folder)
    scenes = args.only or mod.SCENES
    flag, _ = QUALITY[args.quality]
    media = folder / "media"
    video_dir = folder / "video"
    video_dir.mkdir(exist_ok=True)

    for name in scenes:
        cmd = [
            sys.executable, "-m", "manim", flag, "--frame_rate", str(args.fps),
            "--media_dir", str(media), "--disable_caching", "-v", "WARNING",
            str(folder / "scenes.py"), name,
        ]
        print(f"[render] {name}", flush=True)
        subprocess.run(cmd, check=True, cwd=folder)

    def scene_file(name):
        found = sorted((media / "videos" / "scenes").glob(f"*/{name}.mp4"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit(f"missing render for {name}")
        return found[-1]

    if args.sheets:
        sheets = video_dir / "sheets"
        sheets.mkdir(exist_ok=True)
        for name in scenes:
            src = scene_file(name)
            step = max(duration(src) / 12, 0.5)
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
                 f"fps=1/{step:.2f},scale=640:-1,tile=3x4:padding=6:color=white", "-frames:v", "1",
                 str(sheets / f"{name}.png")],
                check=True,
            )
        print(f"[sheets] {sheets}")

    if args.only:
        return

    # Each scene's audio ends at its last sound, before the video ends. Pad it to the
    # video length, or the narration drifts earlier with every scene after joining.
    padded = video_dir / "padded"
    padded.mkdir(exist_ok=True)
    files = []
    for n in mod.SCENES:
        src = scene_file(n)
        dst = padded / src.name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", f"apad=whole_dur={duration(src):.3f}",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(dst)],
            check=True,
        )
        files.append(dst)
    listing = video_dir / "concat.txt"
    listing.write_text("".join(f"file '{f}'\n" for f in files))
    out = video_dir / f"{folder.name}.mp4"
    joined = video_dir / "joined.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-af", "aresample=async=1", "-c:a", "aac", "-b:a", "192k", str(joined)],
        check=True,
    )
    if args.no_music:
        joined.replace(out)
    else:
        sys.path.insert(0, str(ROOT))
        from common import audio

        bed = audio.music(duration(joined))
        # Music ducks under the voice (sidechain), then both are mixed.
        graph = (
            f"[1:a]volume={args.music_db}dB[m];"
            "[0:a]asplit=2[v1][v2];"
            "[m][v1]sidechaincompress=threshold=0.02:ratio=5:attack=30:release=600[duck];"
            "[v2][duck]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[a]"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(joined), "-i", str(bed), "-filter_complex", graph,
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", str(out)],
            check=True,
        )
        joined.unlink()
    chapters, t = [], 0.0
    titles = getattr(mod, "CHAPTERS", {})
    for name, f in zip(mod.SCENES, files):
        chapters.append({"scene": name, "title": titles.get(name, name), "start": round(t, 2)})
        t += duration(f)
    (video_dir / "chapters.json").write_text(json.dumps(chapters, indent=2))
    # index.html is opened from disk, where fetch() of local JSON is blocked; a script tag works.
    (video_dir / "chapters.js").write_text(f"window.CHAPTERS = {json.dumps(chapters)};\n")
    listing.unlink()
    for f in files:
        f.unlink()
    padded.rmdir()
    print(f"[done] {out}  ({t:.0f} s)")


if __name__ == "__main__":
    main()
