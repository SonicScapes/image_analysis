#!/usr/bin/env python3
"""
scene_index.py — list the scenes for the viewer.

A browser cannot list a directory over HTTP, so the app needs a file telling it which
scenes exist. Every tool that creates or changes a scene calls write_index(); you can
also run this directly after editing things by hand.

    python src/python/scene_index.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = REPO / "data" / "scenes"


def write_index(scenes_dir=DEFAULT_SCENES, quiet=True):
    scenes_dir = Path(scenes_dir)
    entries = []
    for d in sorted(p for p in scenes_dir.iterdir() if p.is_dir()):
        scene_file = d / "scene.json"
        if not scene_file.exists():
            continue
        try:
            s = json.loads(scene_file.read_text())
        except Exception:
            continue
        pano = s.get("panorama", {})
        entries.append({
            "id": d.name,
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
