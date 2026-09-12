#!/usr/bin/env python3
"""
check_coverage.py — do we have a sound for everything we can see?

Reads each scene's `<image>_classes.txt` (what segmentation found) and its `scene.json`
(what audio is actually wired up), and reports the gap. A class that covers a quarter of
the view with no sound attached is the most audible hole a scene can have, and it is
invisible until you look for it.

    python check_coverage.py                        # every scene under scenes/
    python check_coverage.py --scene poi-1
    python check_coverage.py --queries              # print Freesound queries for the gaps

Classes whose SOUND_SPEC type is None (sky, trail) are recognised but deliberately
silent — the wind bed covers the sky — so they are listed as such, not as gaps.

Scenes live in category subfolders (scenes/360pano/<id>, scenes/photos/<id>), so scenes
are discovered by walking for `*_classes.txt` rather than listing the root directly.
"""

import argparse, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SCENES = REPO / "scenes"

# Reuse the single source of truth rather than restating it here.
_spec = importlib.util.spec_from_file_location(
    "segpan", REPO / "src" / "python" / "image_analysis" / "segment_panorama.py")
_segpan = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_segpan)
    SOUND_SPEC, QUERIES = _segpan.SOUND_SPEC, _segpan.QUERIES
except Exception as e:                                   # torch missing, etc.
    print(f"note: could not import segment_panorama ({e}); using a built-in copy",
          file=sys.stderr)
    SOUND_SPEC, QUERIES = {}, {}

# A layer id is not always its class: prepare_audio.py may emit water-2, rock-3, and
# fetch-sounds.mjs names things after the query. Strip trailing -N and map aliases.
ALIASES = {
    "birds": "animal", "cowbell": "cattle", "infrastructure": "cablecar",
    "voices": "person", "footsteps": "person", "bed": "wind", "lake": "water",
    "stream": "water", "wind": "rock",   # a wind bed is what a rock face sounds like
}


def layer_classes(scene):
    out = set()
    for l in scene.get("layers", []):
        name = l.get("class") or l["id"]
        name = name.rsplit("-", 1)[0] if name.rsplit("-", 1)[-1].isdigit() else name
        out.add(ALIASES.get(name, name))
        out.add(name)
    return out


def read_classes(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[-1].endswith("%"):
            try:
                rows.append((" ".join(parts[:-1]), int(parts[-1].rstrip("%"))))
            except ValueError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(DEFAULT_SCENES))
    ap.add_argument("--scene", default=None, help="just this one")
    ap.add_argument("--queries", action="store_true",
                    help="print a Freesound query per gap, ready to paste")
    ap.add_argument("--json", dest="json_out", nargs="?", const="scenes/audio-needs.json",
                    default=None,
                    help="write the gaps as JSON for fetch-sounds.mjs to consume "
                         "(default path: scenes/audio-needs.json)")
    ap.add_argument("--min-percent", type=int, default=1,
                    help="ignore classes below this share (default %(default)s)")
    args = ap.parse_args()

    root = Path(args.scenes)
    all_dirs = sorted({p.parent for p in root.rglob("*_classes.txt")
                       if p.parent.relative_to(root).parts[0] != "audio"})
    dirs = [d for d in all_dirs if d.name == args.scene] if args.scene else all_dirs

    all_gaps, gap_scenes = {}, {}
    for d in dirs:
        # The report is named after the source image (IMG_1234_classes.txt); older
        # scenes used a fixed name, so accept both.
        reports = sorted(d.glob("*_classes.txt"))
        scene_file = d / "scene.json"
        if not reports:
            continue
        cls_file = reports[0]
        detected = [(c, v) for c, v in read_classes(cls_file) if v >= args.min_percent]
        have = layer_classes(json.loads(scene_file.read_text())) if scene_file.exists() else set()

        print(f"\n{d.name}")
        covered_share = gap_share = 0
        for cls, pct in detected:
            if cls == "other":
                print(f"  {pct:>3}%  {cls:<12} unmodelled — no category, no sound needed")
                continue
            spec = SOUND_SPEC.get(cls)
            if spec and spec[0] is None:
                print(f"  {pct:>3}%  {cls:<12} silent by design")
                continue
            if cls in have:
                covered_share += pct
                print(f"  {pct:>3}%  {cls:<12} OK — audio wired up")
            else:
                gap_share += pct
                kind = spec[0] if spec else "unknown class"
                print(f"  {pct:>3}%  {cls:<12} MISSING ({kind})")
                all_gaps[cls] = max(all_gaps.get(cls, 0), pct)
                gap_scenes.setdefault(cls, []).append(d.name)
        print(f"        {covered_share}% of the view has sound, {gap_share}% does not")

    if all_gaps:
        print(f"\n{len(all_gaps)} class(es) missing audio across all scenes, "
              "biggest share first:")
        for cls, pct in sorted(all_gaps.items(), key=lambda t: -t[1]):
            q = QUERIES.get(cls, cls)
            print(f"  {pct:>3}%  {cls:<12} {q!r}")
        if args.queries:
            print("\nAdd these to the SCENE table in src/js/tools/fetch-sounds.mjs:")
            for cls, pct in sorted(all_gaps.items(), key=lambda t: -t[1]):
                spec = SOUND_SPEC.get(cls, ("region", -8, 8, 200, 0.02))
                print(f"  {{ id: '{cls}', type: '{spec[0]}', query: '{QUERIES.get(cls, cls)}', "
                      f"az: 0, el: 0, spread: 50, distance: {spec[3]}, gain: {spec[1]} }},")
    else:
        print("\nEvery sounding class in every scene has audio attached.")

    if args.json_out:
        # The class taxonomy lives in Python. Export what is missing so the JavaScript
        # fetcher does not have to keep its own copy of it and drift.
        out = Path(args.json_out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        needs = []
        for cls, pct in sorted(all_gaps.items(), key=lambda t: -t[1]):
            spec = SOUND_SPEC.get(cls, ("region", -8, 8, 200, 0.02))
            kind = spec[0] or "region"
            needs.append({
                "class": cls,
                "type": kind,
                "query": QUERIES.get(cls, cls),
                "max_share_percent": pct,
                "scenes": sorted(set(gap_scenes.get(cls, []))),
                "gain": spec[1],
                "distance": spec[3],
                # Events want several takes so the scheduler can vary them; a region
                # only ever plays one loop.
                "count": 3 if kind == "event" else 1,
                "max_seconds": 8 if kind == "event" else 90,
                "min_seconds": 0 if kind == "event" else 15,
            })
        out.write_text(json.dumps({"needs": needs}, indent=2))
        print(f"\nwrote {out} ({len(needs)} class(es) to fetch)")
        print("  node src/js/tools/fetch-sounds.mjs")


if __name__ == "__main__":
    main()
