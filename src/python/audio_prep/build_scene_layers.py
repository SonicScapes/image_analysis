#!/usr/bin/env python3
"""
build_scene_layers.py — give each scene the audio its own image asks for.

This is the join between the two halves of the pipeline. Until now they have run past
each other: segmentation measured WHERE things are and produced `scene.segmented.json`
with real azimuths and distances but placeholder filenames, while `prepare_audio.py`
produced real audio with guessed geometry from a filename hint table. Neither file on
its own is a scene you would want to listen to.

This reads both, plus whatever is in the shared pool, and writes the layers into
`scene.json`:

    geometry  <- scene.segmented.json   (measured from the photograph)
    audio     <- data/audio/            (our recordings first, downloads second)

    python build_scene_layers.py                    # every scene
    python build_scene_layers.py --scene hohe-tauern --dry-run

A layer you have tuned by ear is not overwritten if you mark it `"locked": true`.
Everything else is regenerated, so re-running after a better segmentation pass or a new
download simply improves the scene.
"""

import argparse, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/python
from paths import SCENES_DIR, AUDIO_DIR

DEFAULT_SCENES = SCENES_DIR
DEFAULT_AUDIO = AUDIO_DIR

# A stem recorded for one thing can legitimately serve a class named for another: our
# footsteps take is what `person` sounds like from two metres, and a wind bed is what an
# exposed rock face sounds like. Keys are classes, values are stems to accept for them.
STEM_ALIASES = {
    "person": ["person", "voices", "footsteps"],
    "rock": ["rock", "wind"],
    "scree": ["scree", "rock", "wind"],
    "snow": ["snow", "wind"],
    "cattle": ["cattle", "cowbell"],
    "animal": ["animal", "birds"],
    "cablecar": ["cablecar", "infrastructure", "lift"],
    "water": ["water", "stream", "lake"],
    "waterfall": ["waterfall", "water"],
}


def pool_index(audio_dir):
    """class -> [relative filenames], newest naming first (water.mp3 before water-2.mp3)."""
    out = {}
    for f in sorted(audio_dir.glob("*.mp3")):
        base = re.sub(r"-\d+$", "", f.stem)
        out.setdefault(base, []).append(f.name)
    return out


def stems_for(cls, pool):
    for candidate in STEM_ALIASES.get(cls, [cls]):
        if candidate in pool:
            return candidate, pool[candidate]
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(DEFAULT_SCENES))
    ap.add_argument("--audio", default=str(DEFAULT_AUDIO))
    ap.add_argument("--scene", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-share", type=int, default=0,
                    help="skip classes covering less than this %% of the view")
    args = ap.parse_args()

    root, audio_dir = Path(args.scenes), Path(args.audio)
    if not audio_dir.is_dir():
        raise SystemExit(f"no audio pool at {audio_dir} — run prepare_audio.py first")
    pool = pool_index(audio_dir)
    if not pool:
        raise SystemExit(f"{audio_dir} is empty — run prepare_audio.py first")
    print(f"pool: {len(pool)} class(es) — {', '.join(sorted(pool))}\n")

    dirs = ([root / args.scene] if args.scene
            else sorted(d for d in root.iterdir() if d.is_dir()))
    gaps_all = {}

    for d in dirs:
        seg_file, scene_file = d / "scene.segmented.json", d / "scene.json"
        if not seg_file.exists() or not scene_file.exists():
            continue
        seg = json.loads(seg_file.read_text())
        scene = json.loads(scene_file.read_text())
        shares = seg.get("classes", {})

        locked = {l["id"]: l for l in scene.get("layers", []) if l.get("locked")}
        built, gaps = [], []

        for region in seg.get("layers", []):
            cls = re.sub(r"-\d+$", "", region["id"])
            share = shares.get(cls, 0)
            if share < args.min_share:
                continue
            stem_cls, files = stems_for(cls, pool)
            if not files:
                gaps.append((cls, share))
                continue

            layer = {
                "id": region["id"],
                "class": cls,
                "type": region.get("type", "region"),
                # Geometry from the image, not from a filename guess.
                "az": region["az"], "el": region["el"],
                "spread": region["spread"], "distance": region["distance"],
                "gain": region.get("gain", -8), "focus": region.get("focus", 8),
                # Events take every variant; a region plays one loop.
                "src": ([f"../../audio/{f}" for f in files]
                        if region.get("type") == "event" and len(files) > 1
                        else f"../../audio/{files[0]}"),
            }
            if stem_cls != cls:
                layer["borrowed_from"] = stem_cls
            built.append(layer)

        # Anything the user locked survives untouched, and wins on id collisions.
        final = [l for l in built if l["id"] not in locked] + list(locked.values())
        final.sort(key=lambda l: {"bed": 0, "region": 1, "event": 2}.get(l["type"], 1))

        covered = sum(shares.get(re.sub(r"-\d+$", "", l["id"]), 0) for l in final)
        print(f"{d.name}: {len(final)} layer(s), ~{covered}% of the view has sound"
              + (f", {len(locked)} locked" if locked else ""))
        for l in final:
            src = l["src"] if isinstance(l["src"], str) else f"[{len(l['src'])} variants]"
            borrowed = f"  (borrowed {l['borrowed_from']})" if l.get("borrowed_from") else ""
            print(f"    {l['type']:<7} {l['id']:<14} az {l['az']:>5.0f}  "
                  f"{l['distance']:>4} m  {Path(str(src)).name}{borrowed}")
        for cls, share in sorted(gaps, key=lambda t: -t[1]):
            print(f"    {'--':<7} {cls:<14} {share:>3}% of the view — NO AUDIO")
            gaps_all[cls] = max(gaps_all.get(cls, 0), share)

        if not args.dry_run:
            scene["layers"] = final
            scene_file.write_text(json.dumps(scene, indent=2))

    if gaps_all:
        print(f"\nStill missing audio for {len(gaps_all)} class(es): "
              + ", ".join(f"{c} ({p}%)" for c, p in
                          sorted(gaps_all.items(), key=lambda t: -t[1])))
        print("  python src/python/audio_prep/check_coverage.py --json")
        print("  node src/js/tools/fetch-sounds.mjs")
        print("  python src/python/audio_prep/prepare_audio.py "
              "--in resources/sounds-freesound --scene <any>")
    if args.dry_run:
        print("\n(dry run — nothing written)")
    else:
        from scene_index import write_index
        dest, entries = write_index(root)
        print(f"\nindex: {len(entries)} scene(s) -> {dest}")


if __name__ == "__main__":
    main()
