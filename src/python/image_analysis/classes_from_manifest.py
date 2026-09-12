#!/usr/bin/env python3
"""
classes_from_manifest.py — rebuild the class report from an existing scene.segmented.json.

For scenes segmented before the class report existed, or whenever you want the numbers
again without spending another twenty inference passes. Reads the `grid` block that every
segmented manifest carries and re-derives the coverage from it, in about a second.

    python classes_from_manifest.py                       # every scene under data/scenes
    python classes_from_manifest.py --scene hohe-tauern

ONE DIFFERENCE from the report `segment_panorama.py` writes: this works from the label
grid, which is an argmax — one winning class per cell — so the probability mass the model
gave to classes we do not model is gone and `other` cannot be recovered. What you get is
"share of the view labelled X", which is what `labels.png` shows. The generated file says
so in its header rather than quietly implying the two are the same.
"""

import argparse, json, math, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/python
from paths import SCENES_DIR

DEFAULT_SCENES = SCENES_DIR
VOID = 255


def shares_from_grid(grid):
    """Solid-angle weighted share per class, from the argmax label grid."""
    gw, gh = grid["w"], grid["h"]
    names = grid["classes"]
    cells = np.array(grid["cells"], dtype=np.int32).reshape(gh, gw)

    # cos(latitude) weighting, exactly as the region extractor uses: an equirect grid
    # devotes as many cells to the poles as to the horizon, and the poles are a handful
    # of real steradians.
    lat = (0.5 - (np.arange(gh) + 0.5) / gh) * math.pi
    cell = (np.cos(lat) * (math.pi / gh) * (2 * math.pi / gw))[:, None] * np.ones((1, gw))

    photographed = cell[cells != VOID].sum()
    if photographed <= 0:
        return []
    out = []
    for i, name in enumerate(names):
        w = cell[cells == i].sum()
        if w > 0:
            out.append((name, w / photographed))
    out.sort(key=lambda t: -t[1])
    return out


def as_percentages(shares, floor=1):
    """Integer percentages summing to 100, by largest remainder."""
    keep = [(n, v) for n, v in shares if v * 100 >= floor]
    total = sum(v for _, v in keep) or 1.0
    scaled = [(n, v / total * 100) for n, v in keep]
    ints = [[n, int(math.floor(x))] for n, x in scaled]
    remainder = 100 - sum(i for _, i in ints)
    order = sorted(range(len(scaled)), key=lambda k: -(scaled[k][1] - ints[k][1]))
    for k in order[:max(remainder, 0)]:
        ints[k][1] += 1
    rows = [(n, v) for n, v in ints if v > 0]
    rows.sort(key=lambda t: (t[0] == "other", -t[1]))
    return rows


def sky_sanity(cells, gh, gw, names):
    """Warn when the upper hemisphere of a 360 contains almost no sky — see
    segment_panorama.sky_sanity. Duplicated here so this tool stays importable without
    torch installed."""
    if "sky" not in names:
        return None
    sky_i = names.index("sky")
    lat = (0.5 - (np.arange(gh) + 0.5) / gh) * math.pi
    cell = (np.cos(lat) * (math.pi / gh) * (2 * math.pi / gw))[:, None] * np.ones((1, gw))
    up = lat > 0
    total = cell[up].sum()
    if total <= 0:
        return None
    share = 100 * cell[up][cells[up] == sky_i].sum() / total
    if share >= 20:
        return None
    worst = {n: 100 * cell[up][cells[up] == i].sum() / total
             for i, n in enumerate(names) if cell[up][cells[up] == i].sum() > 0}
    top = sorted(worst.items(), key=lambda t: -t[1])[:2]
    return (f"only {share:.0f}% of the sky is labelled 'sky' — mostly "
            + ", ".join(f"{n} {v:.0f}%" for n, v in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(DEFAULT_SCENES))
    ap.add_argument("--scene", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing report")
    args = ap.parse_args()

    root = Path(args.scenes)
    dirs = ([root / args.scene] if args.scene
            else sorted(d for d in root.iterdir() if d.is_dir()))

    for d in dirs:
        man = d / "scene.segmented.json"
        if not man.exists():
            continue
        data = json.loads(man.read_text())
        grid = data.get("grid")
        if not grid or "cells" not in grid:
            print(f"{d.name}: no grid in the manifest — re-run segment_panorama.py")
            continue

        cells = np.array(grid["cells"], dtype=np.int32).reshape(grid["h"], grid["w"])
        warn = None
        if data.get("panorama", {}).get("hfov_deg", 0) > 350:
            warn = sky_sanity(cells, grid["h"], grid["w"], grid["classes"])
        rows = as_percentages(shares_from_grid(grid))
        pano_block = data.get("panorama", {})
        stem = Path(pano_block.get("source")
                    or pano_block.get("file", "panorama.jpg")).stem
        dest = d / f"{stem}_classes.txt"
        if dest.exists() and not args.force:
            print(f"{d.name}: {dest.name} already exists (use --force)")
            continue

        pano_block = data.get("panorama", {})
        pano = pano_block.get("source") or pano_block.get("file", "panorama.jpg")
        width = max((len(n) for n, _ in rows), default=8)
        lines = [
            f"# class coverage of {pano}",
            "# DERIVED from the label grid in scene.segmented.json, not from a fresh",
            "# segmentation pass. Solid-angle weighted over the photographed area.",
            "# No 'other' row: the grid stores one winning class per cell, so the",
            "# probability the model gave to unmodelled classes is not recoverable here.",
        ]
        if warn:
            src = data.get("source", {})
            lines += [
                "#",
                f"# WARNING: {warn}",
                "# A full 360 is about half sky, so this segmentation is wrong and these",
                "# numbers describe a bad run. Re-run with --mode multiview and the B4",
                f"# default model (this was mode={src.get('mode')}, {src.get('model')}).",
            ]
        lines += ["", ] + [f"{n:<{width}} {v}%" for n, v in rows]
        dest.write_text("\n".join(lines) + "\n")

        print(f"\n{d.name} -> {dest}")
        for n, v in rows:
            print(f"  {n:<12} {v:>3}%")
        if warn:
            src = data.get("source", {})
            print(f"\n  !! {warn}")
            print(f"  !! this run was mode={src.get('mode')}, {src.get('model')} — the")
            print("  !! smoke-test settings. Re-run with --mode multiview and the B4 default.")

        missing = [c for c in ("snow", "glacier", "cattle", "cablecar", "person")
                   if c not in grid["classes"]]
        if missing:
            print(f"  note: segmented without --open-vocab, so {', '.join(missing)} "
                  "were never looked for")


if __name__ == "__main__":
    main()
