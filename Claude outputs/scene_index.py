#!/usr/bin/env python3
"""
scene_index.py — list the scenes for the viewer.

A browser cannot list a directory over HTTP, so the app needs a file telling it which
scenes exist. Every tool that creates or changes a scene calls write_index(); you can
also run this directly after editing things by hand.

    python src/python/scene_index.py

Scenes live in category subfolders under the scenes root (360pano/, photos/ — a photo's
own kind decides which, at ingest time), so this walks the whole tree rather than
assuming one flat layer, and records each scene's `dir` (its path relative to the scenes
root) alongside its `id` (just the folder name) — the app fetches `dir + "/scene.json"`.
One index file for everything, not one per subfolder: the viewer's scene switcher wants
a single flat list to page through, and one file is one thing to keep in sync rather
than several.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = REPO / "scenes"

# Not scene folders, even though they sit under the scenes root.
SKIP_DIRS = {"audio"}


def write_index(scenes_dir=DEFAULT_SCENES, quiet=True):
    scenes_dir = Path(scenes_dir)
    entries = []
    for scene_file in sorted(scenes_dir.rglob("scene.json")):
        d = scene_file.parent
        rel_dir = d.relative_to(scenes_dir)
        if rel_dir.parts and rel_dir.parts[0] in SKIP_DIRS:
            continue
        try:
            s = json.loads(scene_file.read_text())
        except Exception:
            continue
        pano = s.get("panorama", {})
        entries.append({
            "id": d.name,
            "dir": rel_dir.as_posix(),
            "name": s.get("name", d.name),
            "projection": pano.get("projection"),
            "hfov_deg": pano.get("hfov_deg"),
            "layers": len(s.get("layers", [])),
            "segmented": (d / "scene.segmented.json").exists(),
            # Which debug images the viewer can offer for this scene.
            "overlay": (d / "overlay.png").exists(),
            "overlay_labeled": (d / "overlay_labeled.png").exists(),
            "labels": (d / "labels.png").exists(),
            "elev_m": pano.get("elev_m"),
            "lat": pano.get("lat"),
            "lon": pano.get("lon"),
        })
    dest = scenes_dir / "index.json"
    dest.write_text(json.dumps({"scenes": entries}, indent=2))
    if not quiet:
        print(f"wrote {dest} ({len(entries)} scene(s), "
              f"{sum(1 for e in entries if e['layers'])} with audio)")
    return dest, entries


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCENES
    write_index(target, quiet=False)
