#!/usr/bin/env python3
"""
prepare_audio.py — turn the handheld-recorder WAVs into web-ready scene assets.

Your own recordings are the most valuable material in this project: nobody else at the
hackathon has the actual sound of the place. But raw 48 kHz WAVs are 20-50 MB each and
don't loop, so they can't go into a browser as they are.

This does the whole conversion:

  1. Finds the most STATIONARY window in each recording. A good ambience loop is one where
     nothing dramatic happens — no gust peak, no footstep, no voice — so we pick the window
     whose short-term loudness varies least. This is the step that makes loops inaudible.
  2. Folds the tail over the head with an equal-power crossfade, so the loop has no seam.
  3. Normalises every stem to the same level, so the engine's gain maths means something.
  4. Encodes to MP3 (and OGG), a few hundred KB per stem instead of tens of MB.
  5. Writes a scene.json skeleton with one layer per recording, ready to tune.

Short files (< 12 s by default) are treated as one-shot events instead: normalised, not
looped, and emitted as `type: "event"`.

    python prepare_audio.py                      # uses the repo's resources/sounds
    python prepare_audio.py --loop-seconds 25 --scene hohe-tauern-wasserfall

Needs: ffmpeg + ffprobe on PATH, numpy. Nothing else.
"""

import argparse, json, math, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_IN = REPO / "resources" / "sounds"
DEFAULT_OUT = REPO / "data" / "scenes"

SR = 48000              # working sample rate
ANALYSIS_SR = 8000      # enough to find a stationary window, 6x faster to decode
TARGET_RMS_DB = -23.0   # every loop lands here; the engine's per-layer gain does the rest
TARGET_PEAK_DB = -1.0


def rel(p):
    """Path relative to the repo when it is inside it, absolute otherwise."""
    try:
        return Path(p).relative_to(REPO)
    except ValueError:
        return Path(p)

# Filename hints -> sound class, so the scene skeleton comes out roughly right.
# Add your own as you label takes in the field; the guessing is a convenience, not magic.
HINTS = [
    (r"waterfall|wasserfall",  "waterfall", dict(az=285, el=-10, spread=30, distance=110, gain=2,  focus=10)),
    (r"water|stream|bach|creek","water",    dict(az=240, el=-25, spread=45, distance=35,  gain=-4, focus=9)),
    (r"wind|sturm",            "wind",      dict(az=320, el=15,  spread=90, distance=30,  gain=-10, focus=5, tags=["wind"])),
    (r"forest|wald|tree",      "forest",    dict(az=150, el=-5,  spread=55, distance=180, gain=-6, focus=8)),
    (r"meadow|wiese|insect",   "pasture",   dict(az=95,  el=-12, spread=60, distance=60,  gain=-12, focus=7)),
    (r"cow|bell|glocke",       "cowbell",   dict(az=110, el=-10, spread=35, distance=300, gain=4,  focus=6)),
    (r"bird|vogel|marmot|chough|raven", "birds",
                                            dict(az=150, el=5,   spread=70, distance=45,  gain=-4, focus=6, tags=["wildlife"])),
    (r"lake|see",              "lake",      dict(az=20,  el=-18, spread=50, distance=260, gain=-8, focus=8)),
    # Human presence. Authored at "busy day" level and held down until the pressure
    # control is raised, so the slider REVEALS them instead of merely turning them up.
    (r"footstep|schritt|step", "footsteps", dict(az=0,   el=-60, spread=40, distance=2,   gain=-8, focus=3, tags=["human"])),
    (r"voice|stimme|people|crowd|talk", "voices",
                                            dict(az=120, el=-8,  spread=50, distance=40,  gain=-6, focus=6, tags=["human"])),
    (r"lift|seilbahn|gondel|cable|road|traffic|car", "infrastructure",
                                            dict(az=60,  el=8,   spread=40, distance=250, gain=-2, focus=8, tags=["human"])),
]


def need(tool):
    if not shutil.which(tool):
        sys.exit(f"{tool} not found on PATH. brew install ffmpeg")


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def decode(path, sr, mono=True, start=None, dur=None):
    """Decode to a numpy float32 array via ffmpeg. Returns (samples, channels)."""
    ch = 1 if mono else 2
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.4f}"]
    cmd += ["-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    return a.reshape(-1, ch) if ch > 1 else a


def most_stationary_window(x, sr, window_s, frame_s=0.25):
    """
    Pick the window whose short-term loudness is most constant — that is what makes a
    loop unnoticeable. Also penalise windows that are near-silent or clipping, because
    a technically perfect loop of nothing is still nothing.
    """
    fl = int(frame_s * sr)
    n = len(x) // fl
    if n < 4:
        return 0.0
    frames = x[:n * fl].reshape(n, fl)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)

    w = max(int(window_s / frame_s), 2)
    if w >= n:
        return 0.0

    best, best_score = 0, math.inf
    for i in range(0, n - w + 1):
        seg = db[i:i + w]
        spread = float(seg.std())
        level = float(seg.mean())
        silence_penalty = max(0.0, -45.0 - level) * 0.8   # quieter than -45 dB is unusable
        loud_penalty = max(0.0, level - (-8.0)) * 1.5     # probably clipping or handling noise
        score = spread + silence_penalty + loud_penalty
        if score < best_score:
            best, best_score = i, score
    return best * frame_s


def normalise(x):
    rms = float(np.sqrt((x ** 2).mean() + 1e-12))
    gain = 10 ** (TARGET_RMS_DB / 20) / rms
    y = x * gain
    peak = float(np.abs(y).max() + 1e-12)
    ceiling = 10 ** (TARGET_PEAK_DB / 20)
    if peak > ceiling:
        y *= ceiling / peak
    return y


def loop_fold(y, sr, crossfade_s=2.0):
    """
    Make the end meet the beginning. Take length N + XF, then crossfade the extra tail
    back over the head with equal-power (sqrt) ramps and return the first N samples.
    """
    xf = int(crossfade_s * sr)
    if len(y) <= xf * 2:
        return y
    n = len(y) - xf
    out = y[:n].copy()
    t = np.linspace(0, 1, xf, dtype=np.float32)[:, None] if y.ndim > 1 else \
        np.linspace(0, 1, xf, dtype=np.float32)
    fade_in, fade_out = np.sqrt(t), np.sqrt(1 - t)
    out[:xf] = out[:xf] * fade_in + y[n:n + xf] * fade_out
    return out


def encode(y, sr, dest, channels):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
           "-ac", str(channels), "-i", "-"]
    if dest.suffix == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", "128k"]
    else:
        cmd += ["-codec:a", "libvorbis", "-q:a", "4"]
    cmd += [str(dest)]
    subprocess.run(cmd, input=y.astype(np.float32).tobytes(), check=True,
                   capture_output=True)


def classify(stem):
    low = stem.lower()
    for pattern, cls, geom in HINTS:
        if re.search(pattern, low):
            return cls, dict(geom)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--scene", default="hohe-tauern")
    ap.add_argument("--loop-seconds", type=float, default=20.0)
    ap.add_argument("--crossfade", type=float, default=2.0)
    ap.add_argument("--event-max-seconds", type=float, default=12.0,
                    help="recordings shorter than this become one-shot events")
    ap.add_argument("--ogg", action="store_true", help="also write .ogg alongside .mp3")
    args = ap.parse_args()

    need("ffmpeg"); need("ffprobe")

    src = Path(args.src)
    out_dir = Path(args.out) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in src.iterdir()
                   if p.suffix.lower() in (".wav", ".aiff", ".aif", ".flac", ".m4a")])
    if not wavs:
        sys.exit(f"no recordings found in {src}")

    print(f"{len(wavs)} recordings -> {out_dir}\n")
    layers, unlabelled, used_names = [], [], set()

    for path in wavs:
        dur = probe_duration(path)
        cls, geom = classify(path.stem)
        is_event = dur <= args.event_max_seconds
        name = cls or path.stem.lower().replace("_", "-")
        # Only avoid collisions WITHIN this run. Colliding with a file from a previous
        # run means the same recording again, so overwrite it — otherwise every re-run
        # silently duplicates every layer as water-2, water-3, ...
        if name in used_names:
            k = 2
            while f"{name}-{k}" in used_names:
                k += 1
            name = f"{name}-{k}"
        used_names.add(name)
        dest = out_dir / f"{name}.mp3"

        if is_event:
            y = decode(path, SR, mono=True)
            y = normalise(y)
            encode(y, SR, dest, 1)
            if args.ogg:
                encode(y, SR, dest.with_suffix(".ogg"), 1)
            kind, extra = "event", dict(ratePerMin=6, jitter=0.7)
            print(f"  {path.name:<28} {dur:6.1f}s  event    -> {dest.name}")
        else:
            mono = decode(path, ANALYSIS_SR, mono=True)
            start = most_stationary_window(mono, ANALYSIS_SR, args.loop_seconds)
            y = decode(path, SR, mono=False, start=start,
                       dur=args.loop_seconds + args.crossfade)
            y = loop_fold(normalise(y), SR, args.crossfade)
            encode(y, SR, dest, 2)
            if args.ogg:
                encode(y, SR, dest.with_suffix(".ogg"), 2)
            kind, extra = ("bed" if name in ("wind", "bed") else "region"), {}
            print(f"  {path.name:<28} {dur:6.1f}s  loop @{start:6.1f}s -> {dest.name} "
                  f"({dest.stat().st_size/1024:.0f} KB)")

        layer = {"id": dest.stem, "type": kind, "src": dest.name}
        if geom:
            layer.update(geom)
        else:
            # Unknown class: park it in front at mid distance and flag it for tuning.
            layer.update(dict(az=0, el=0, spread=50, distance=80, gain=-10, focus=8))
            unlabelled.append(dest.stem)
        layer.update(extra)
        layers.append(layer)

    layers.sort(key=lambda l: {"bed": 0, "region": 1, "event": 2}[l["type"]])

    # Merge into an existing scene.json rather than replacing it: ingest_images.py may
    # already have written the panorama block, and any az/el/distance you have tuned by
    # ear must survive a re-run. Order of the two scripts therefore does not matter.
    scene_path = out_dir / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        existing = {l["id"]: l for l in scene.get("layers", [])}
        merged, kept, added = [], 0, 0
        for layer in layers:
            prev = existing.get(layer["id"])
            if prev:
                # Keep the tuned geometry; only refresh what this script owns.
                prev.update({k: layer[k] for k in ("type", "src")})
                merged.append(prev)
                kept += 1
            else:
                merged.append(layer)
                added += 1
        # Layers that came from somewhere else (hand-written, or fetch-sounds.mjs) stay.
        produced = {l["id"] for l in layers}
        merged += [l for l in scene.get("layers", []) if l["id"] not in produced]
        scene["layers"] = merged
        scene.setdefault("scene", {"scenicness": 8.0, "eventfulness": 6.5})
        note = f"merged into ({kept} layer(s) kept with their tuned geometry, {added} new)"
    else:
        scene = {
            "id": args.scene,
            "name": args.scene.replace("-", " ").title(),
            "north_offset_deg": 0,
            "scene": {"scenicness": 8.0, "eventfulness": 6.5},
            "layers": layers,
        }
        note = "created"
    scene_path.write_text(json.dumps(scene, indent=2))

    print(f"\n{note} {rel(scene_path)}")
    if "panorama" not in scene:
        print("  No panorama yet — run image_analysis/ingest_images.py with the same")
        print(f"  --scene {args.scene} and it will add one.")
    if unlabelled:
        print("\nTune these by ear — they had no filename hint, so they are all at az 0:")
        for u in unlabelled:
            print(f"  {u}")
        print("Rename the source WAVs with a hint (…_waterfall.WAV) and re-run to skip this.")
    print("\nNext: serve the repo root (npm run dev), open src/js/app/index.html,")
    print("and tune az / el / distance in scene.json while listening. Re-running this")
    print("script keeps whatever you tuned.")


if __name__ == "__main__":
    main()
