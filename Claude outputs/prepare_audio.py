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

    python prepare_audio.py                      # our own takes, resources/recordings
    python prepare_audio.py --in resources/sounds-freesound --scene hohe-tauern
    python prepare_audio.py --loop-seconds 25 --scene hohe-tauern-wasserfall

Needs: ffmpeg + ffprobe on PATH, numpy. Nothing else.
"""

import argparse, json, math, os, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# Our own field recordings. (Folder was renamed from `sounds`; fall back to it so a
# teammate who has not pulled still works.)
DEFAULT_IN = (REPO / "resources" / "recordings"
              if (REPO / "resources" / "recordings").is_dir()
              else REPO / "resources" / "sounds")
FREESOUND_IN = REPO / "resources" / "sounds-freesound"
DEFAULT_OUT = REPO / "scenes"
# One shared pool for processed audio. The engine never modifies these files — every
# gain, pan, filter and distance decision happens at runtime from scene.json — so a
# stem is identical for every scene that uses it and copying it per scene is waste.
AUDIO_OUT = REPO / "scenes" / "audio"


def resolve_scene_dir(root, scene_id):
    """
    Scenes live in category subfolders (scenes/360pano/<id>, scenes/photos/<id>), so an
    existing scene is found by name wherever it actually is. A name that matches nothing
    is a brand-new scene with no category yet — put it straight under the root; whatever
    ingests its photo later can move it if it turns out to belong in a subfolder.
    """
    hits = [p for p in root.glob(f"*/{scene_id}") if p.is_dir() and p.parent.name != "audio"]
    return hits[0] if hits else root / scene_id

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

    # Segmentation class names, so files fetched to fill a gap (rock-1.mp3, snow-2.mp3)
    # classify themselves. Geometry here is a starting point; tune it by ear.
    (r"^rock",      "rock",      dict(az=0,   el=5,   spread=70, distance=400, gain=-14, focus=6, tags=["wind"])),
    (r"^snow",      "snow",      dict(az=60,  el=0,   spread=60, distance=300, gain=-16, focus=6)),
    (r"^scree",     "scree",     dict(az=300, el=-10, spread=55, distance=250, gain=-12, focus=7)),
    (r"^glacier",   "glacier",   dict(az=20,  el=0,   spread=40, distance=500, gain=-14, focus=7)),
    (r"^pasture",   "pasture",   dict(az=150, el=-12, spread=60, distance=60,  gain=-12, focus=7)),
    (r"^built",     "built",     dict(az=200, el=-5,  spread=30, distance=150, gain=-18, focus=8)),
    (r"^cattle",    "cattle",    dict(az=150, el=-8,  spread=40, distance=300, gain=4,  focus=6, tags=["wildlife"])),
    (r"^animal",    "animal",    dict(az=330, el=30,  spread=70, distance=250, gain=2,  focus=6, tags=["wildlife"])),
    (r"^marmot",    "marmot",    dict(az=260, el=-5,  spread=40, distance=150, gain=3,  focus=6, tags=["wildlife"])),
    (r"^rockfall",  "rockfall",  dict(az=300, el=0,   spread=45, distance=380, gain=2,  focus=7)),
    (r"^cablecar",  "cablecar",  dict(az=60,  el=8,   spread=40, distance=250, gain=-2, focus=8, tags=["human"])),
    (r"^person",    "person",    dict(az=200, el=-5,  spread=50, distance=60,  gain=-4, focus=7, tags=["human"])),
]


def need(tool):
    """
    Check the tool RUNS, not merely that it exists. A conda ffmpeg with a missing dylib
    is on PATH and aborts with SIGABRT on every call, which surfaces as a traceback from
    whichever file happened to be first — a confusing way to learn your install is broken.
    """
    if not shutil.which(tool):
        sys.exit(f"{tool} not found on PATH.\n"
                 f"  conda install -c conda-forge ffmpeg     (or: brew install ffmpeg)")
    try:
        r = subprocess.run([tool, "-version"], capture_output=True, timeout=20,
                           stdin=subprocess.DEVNULL)
    except Exception as e:
        sys.exit(f"{tool} could not be started: {e}")
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", "replace").strip()
        msg = [f"{tool} is installed but fails to run (exit {r.returncode})."]
        if err:
            msg.append("  " + err.splitlines()[0])
        if "Library not loaded" in err or "dyld" in err:
            msg += ["",
                    "  A dynamic library is missing — a conda ffmpeg build with unmet",
                    "  dependencies. Try, in order:",
                    "    conda install -c conda-forge librsvg",
                    "    conda install -c conda-forge --force-reinstall ffmpeg",
                    "    conda remove --force ffmpeg && brew install ffmpeg"]
        sys.exit("\n".join(msg))


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
        stdin=subprocess.DEVNULL, timeout=120).stdout.strip()
    return float(out)


def decode(path, sr, mono=True, start=None, dur=None):
    """
    Decode to a numpy float32 array via ffmpeg. Returns (samples, channels).

    `-nostdin` matters: ffmpeg reads stdin for interactive keystrokes by default, and
    when it inherits a terminal — which it does when you run this from a shell rather
    than a pipeline — it can sit there waiting instead of decoding. The symptom is a
    hang on the first file with no output at all.
    """
    ch = 1 if mono else 2
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.4f}"]
    cmd += ["-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True,
                         stdin=subprocess.DEVNULL, timeout=600).stdout
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
    # No -nostdin here: stdin IS the audio stream ("-i -").
    subprocess.run(cmd, input=y.astype(np.float32).tobytes(), check=True,
                   capture_output=True, timeout=600)


def geom_for_class(cls):
    """Starting geometry for a class name, from the same table the filename hints use."""
    for _pattern, name, geom in HINTS:
        if name == cls:
            return dict(geom)
    return None


def classify(stem, labels=None):
    """
    An explicit label wins over a filename guess. survey_recordings.py writes
    labels.json by measuring what is actually in each take, which beats hoping somebody
    renamed R07_0021.WAV in the field.
    """
    if labels:
        entry = labels.get(stem) or labels.get(f"{stem}.WAV") or labels.get(f"{stem}.wav")
        if entry:
            cls = entry.get("class") if isinstance(entry, dict) else entry
            if cls:
                return cls, (geom_for_class(cls) or
                             dict(az=0, el=0, spread=50, distance=80, gain=-10, focus=8))
    low = stem.lower()
    for pattern, cls, geom in HINTS:
        if re.search(pattern, low):
            return cls, dict(geom)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--audio-out", default=str(AUDIO_OUT),
                    help="shared folder for processed audio (default %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="re-encode stems that already exist in the shared pool")
    ap.add_argument("--scene", default="hohe-tauern")
    ap.add_argument("--pick", nargs="*", default=None,
                    help="only these recordings (filenames or glob patterns). Without it "
                         "EVERY file in --in becomes a layer, which for two dozen takes "
                         "means two dozen loops playing at once.")
    ap.add_argument("--labels", default=None,
                    help="JSON mapping filename -> class, from survey_recordings.py "
                         "(default: labels.json in the input folder, if present)")
    ap.add_argument("--include-unusable", action="store_true",
                    help="process takes labels.json marked unusable")
    ap.add_argument("--labelled-only", action="store_true",
                    help="use only recordings whose filename matches a hint (…_water, "
                         "…_waterfall, …_footsteps). The fast way to a clean first mix.")
    ap.add_argument("--loop-seconds", type=float, default=20.0)
    ap.add_argument("--crossfade", type=float, default=2.0)
    ap.add_argument("--event-max-seconds", type=float, default=12.0,
                    help="recordings shorter than this become one-shot events")
    ap.add_argument("--ogg", action="store_true", help="also write .ogg alongside .mp3")
    args = ap.parse_args()

    need("ffmpeg"); need("ffprobe")

    labels = {}
    labels_file = Path(args.labels) if args.labels else Path(args.src) / "labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text())
        print(f"using {rel(labels_file)} ({len(labels)} labelled take(s))\n")

    src = Path(args.src)
    out_dir = resolve_scene_dir(Path(args.out), args.scene)   # scene.json lives here
    audio_dir = Path(args.audio_out)               # the stems live here, shared
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in src.iterdir()
                   if p.suffix.lower() in (".wav", ".aiff", ".aif", ".flac", ".m4a",
                                           ".mp3", ".ogg", ".opus")])
    if args.pick:
        import fnmatch
        chosen = []
        for p_ in wavs:
            if any(fnmatch.fnmatch(p_.name, pat) or p_.name == pat for pat in args.pick):
                chosen.append(p_)
        wavs = chosen
    if args.labelled_only:
        wavs = [p_ for p_ in wavs if classify(p_.stem, labels)[0] is not None]
    if not wavs:
        sys.exit(f"no recordings selected in {src}")
    if len(wavs) > 12 and not (args.pick or args.labelled_only):
        print(f"note: {len(wavs)} recordings will become {len(wavs)} simultaneous layers.")
        print("      Use --labelled-only, or --pick, to start with a mix you can hear.\n")

    print(f"{len(wavs)} recordings -> {rel(audio_dir)}  (scene: {args.scene})\n")
    layers, unlabelled, used_names = [], [], set()

    skipped = []
    for path in wavs:
        try:
            dur = probe_duration(path)
        except Exception as e:
            skipped.append((path.name, f"unreadable ({type(e).__name__})"))
            print(f"  {path.name:<28} SKIPPED — could not read duration")
            continue
        print(f"  {path.name:<28} {dur:6.1f}s  …", end="\r", flush=True)
        entry = labels.get(path.name) if labels else None
        if isinstance(entry, dict) and entry.get("usable") is False and not args.include_unusable:
            skipped.append((path.name, f"labelled unusable ({entry.get('guess', '?')})"))
            print(f"  {path.name:<28} {dur:6.1f}s  skipped — labelled unusable")
            continue
        cls, geom = classify(path.stem, labels)
        is_event = dur <= args.event_max_seconds
        name = cls or path.stem.lower().replace("_", "-")
        # Only avoid collisions WITHIN this run. Colliding with a file from a previous
        # run means the same recording again, so overwrite it — otherwise every re-run
        # silently duplicates every layer as water-2, water-3, ...
        base_class = name          # before any -2 suffix: this is what groups takes
        if name in used_names:
            k = 2
            while f"{name}-{k}" in used_names:
                k += 1
            name = f"{name}-{k}"
        used_names.add(name)
        dest = audio_dir / f"{name}.mp3"
        reuse = dest.exists() and not args.force

        try:
            if reuse:
                kind = ("event" if is_event else
                        "bed" if name in ("wind", "bed") else "region")
                extra = dict(ratePerMin=6, jitter=0.7) if is_event else {}
                print(f"  {path.name:<28} {dur:6.1f}s  reused   -> {dest.name}")
            elif is_event:
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
        except Exception as e:
            skipped.append((path.name, f"conversion failed ({type(e).__name__})"))
            print(f"  {path.name:<28} {dur:6.1f}s  SKIPPED — conversion failed")
            used_names.discard(name)
            continue

        # Relative from the scene folder to the shared pool, so it resolves the same
        # whether the app is served from the repo root or anywhere else.
        rel_src = os.path.relpath(dest, out_dir).replace(os.sep, "/")
        layer = {"id": dest.stem, "type": kind, "src": rel_src,
                 "_class": base_class, "_dur": dur}
        if geom:
            layer.update(geom)
        else:
            # Unknown class: park it in front at mid distance and flag it for tuning.
            layer.update(dict(az=0, el=0, spread=50, distance=80, gain=-10, focus=8))
            unlabelled.append(dest.stem)
        layer.update(extra)
        layers.append(layer)

    if not layers:
        sys.exit("\nNo recordings could be converted — nothing written. "
                 "Check the errors above.")

    # Several takes of the same thing are not several sources.
    #
    # For a REGION that would be four footstep loops playing at once from the same
    # direction — mud, and 6 dB too loud. Keep the longest take; the others stay in the
    # pool, unused, in case you prefer one by ear.
    #
    # For an EVENT the opposite is true: variants are what stop a cowbell sounding like
    # a sample. They collapse into ONE layer whose scheduler picks among them, which is
    # what the engine's `src` array is for.
    grouped, extras = {}, []
    for l in layers:
        cls = l.pop("_class")
        dur = l.pop("_dur")
        g = grouped.setdefault(cls, [])
        g.append((dur, l))
    merged = []
    for cls, items in grouped.items():
        items.sort(key=lambda t: -t[0])
        first = items[0][1]
        if first["type"] == "event" and len(items) > 1:
            first["src"] = [i[1]["src"] for i in items]
            first["id"] = cls
            merged.append(first)
            print(f"  {cls}: {len(items)} variants in one event layer")
        else:
            first["id"] = cls
            merged.append(first)
            for _d, other in items[1:]:
                extras.append(other["id"])
    layers = merged
    if extras:
        print(f"\n{len(extras)} extra take(s) kept in the pool but not wired up "
              f"(one loop per region is enough): " + ", ".join(extras[:6]))
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
    print(f"audio pool: {rel(audio_dir)} "
          f"({len(list(audio_dir.glob('*.mp3')))} stem(s) shared across all scenes)")
    if "panorama" not in scene:
        print("  No panorama yet — run image_analysis/ingest_images.py with the same")
        print(f"  --scene {args.scene} and it will add one.")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for n, why in skipped:
            print(f"  {n:<28} {why}")
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
