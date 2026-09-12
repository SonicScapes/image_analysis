#!/usr/bin/env python3
"""
regen_overlays.py — rebuild overlay.png and overlay_labeled.png as small RGBA tints.

The problem: the original overlay.png blended the label colours INTO the photograph
(0.62 * photo + 0.38 * colour) and saved that as a lossless PNG. That bakes a full copy
of the photograph's high-frequency detail into the file, which is exactly what PNG
compresses worst — so a 6144x3072 overlay came out at ~20 MB, twelve times the photo
it was drawn over.

The photograph is already in the browser. All the overlay needs to carry is the tint:
a transparent PNG, opaque only over classified ground, that the page layers on top of
the same <img> with position:absolute. Flat colour + a mostly-transparent alpha channel
is precisely what PNG compresses BEST, so the same picture drops to tens or hundreds of
kilobytes instead of tens of megabytes.

This does not re-run the segmentation model. labels.png already holds every pixel's
class as a flat colour, and segment_panorama.py always draws it as an equirectangular
map (a full 360x180 sphere, however little of it the photo actually covers) — so any
photo's own per-pixel direction (via its own projection/hfov/vfov) tells us where to
sample it. That is the same sampling `direction_to_pixel` uses for the text labels, just
run per-pixel instead of per-region, and it costs a resize and a lookup, not a model.

    python regen_overlays.py                       # every scene, in place
    python regen_overlays.py --scene poi-1 --keep   # one scene, old files kept as .bak
"""

import argparse, json, math, re, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_panorama import (flat_view, direction_to_pixel, _load_font, PALETTE,
                               TINT_ALPHA, VOID_RGB)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SCENES = REPO / "data" / "scenes"

VOID = np.array(VOID_RGB, np.uint8)
ALPHA = TINT_ALPHA                           # same opacity segment_panorama.py now uses
MAX_LABELS = 12


def sphere_lonlat(w, h, projection, hfov, vfov):
    """
    Longitude/latitude (radians) for every pixel of THIS photo, in the photo's own
    pixel grid — the inverse of how segment_panorama.py projected the sphere onto it.
    """
    if projection == "flat":
        _, lon, lat = flat_view(np.zeros((h, w, 4), np.uint8), hfov, vfov)
        return lon, lat
    # equirect and cylindrical are both linear in lon/lat, differing only in how much
    # of the sphere the frame covers — a full 360 or a partial sweep.
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    lon = math.radians(hfov) * (j / w - 0.5)
    lat = math.radians(vfov) * (0.5 - i / h)
    return lon, lat


def tint_from_labels(labels_png, w, h, projection, hfov, vfov):
    """The RGBA tint for one photo, sampled from its labels.png sphere texture."""
    lon, lat = sphere_lonlat(w, h, projection, hfov, vfov)
    gw, gh = labels_png.size
    arr = np.asarray(labels_png.convert("RGB"))
    gx = np.clip(((lon / (2 * math.pi) + 0.5) * gw).astype(int), 0, gw - 1)
    gy = np.clip(((0.5 - lat / math.pi) * gh).astype(int), 0, gh - 1)
    rgb = arr[gy, gx]
    void = np.all(rgb == VOID, axis=-1)
    alpha = np.where(void, 0, ALPHA).astype(np.uint8)
    return np.dstack([rgb, alpha])


def draw_labels(tint_rgba, regions, shares, projection, hfov, vfov):
    """The same text + centroid dots the original overlay_labeled.png carried, on a
    transparent canvas instead of a photo-blended one."""
    from PIL import ImageDraw
    img = Image.fromarray(tint_rgba, mode="RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(14, int(w / 70))
    font = _load_font(size)

    for r in sorted(regions, key=lambda r: -r.get("solid_angle_sr", 0))[:MAX_LABELS]:
        cls = re.sub(r"-\d+$", "", r["id"])
        pos = direction_to_pixel(r["az"], r["el"], w, h, projection, hfov, vfov)
        if pos is None:
            continue
        x, y = pos
        if not (-w < x < 2 * w and 0 <= y <= h):
            continue
        x = min(max(x, size), w - size)
        y = min(max(y, size), h - size)

        colour = PALETTE.get(cls, (200, 200, 200))
        share = shares.get(cls)
        text = r["id"] if share is None else f"{r['id']}  {share}%"
        text += f"\n{r['distance']} m"

        box = draw.multiline_textbbox((x, y), text, font=font, anchor="mm", spacing=2)
        pad = size * 0.35
        draw.rounded_rectangle(
            [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
            radius=size * 0.3, fill=(8, 12, 11, 205), outline=colour + (255,),
            width=max(2, size // 12))
        draw.multiline_text((x, y), text, font=font, fill=(240, 246, 242, 255),
                            anchor="mm", align="center", spacing=2)
        rr = max(3, size // 6)
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=colour + (255,))
    return img


def regen_one(scene_dir, keep=False):
    scene_file = scene_dir / "scene.json"
    seg_file = scene_dir / "scene.segmented.json"
    labels_file = scene_dir / "labels.png"
    if not (scene_file.exists() and seg_file.exists() and labels_file.exists()):
        print(f"{scene_dir.name}: skipped (missing scene.json / scene.segmented.json / labels.png)")
        return None

    scene = json.loads(scene_file.read_text())
    seg = json.loads(seg_file.read_text())
    pano = scene["panorama"]
    w, h = pano.get("width"), pano.get("height")
    if not (w and h):
        with Image.open(scene_dir / pano["file"]) as im:
            w, h = im.size
    projection, hfov, vfov = pano["projection"], pano["hfov_deg"], pano["vfov_deg"]

    labels_png = Image.open(labels_file)
    tint = tint_from_labels(labels_png, w, h, projection, hfov, vfov)

    before = []
    after = []

    old = scene_dir / "overlay.png"
    if old.exists():
        before.append(old.stat().st_size)
        if keep:
            old.rename(scene_dir / "overlay.png.bak")
    Image.fromarray(tint, mode="RGBA").save(scene_dir / "overlay.png", optimize=True)
    after.append((scene_dir / "overlay.png").stat().st_size)

    old = scene_dir / "overlay_labeled.png"
    if old.exists():
        before.append(old.stat().st_size)
        if keep:
            old.rename(scene_dir / "overlay_labeled.png.bak")
    labeled = draw_labels(tint, seg.get("layers", []), seg.get("classes", {}),
                          projection, hfov, vfov)
    labeled.save(scene_dir / "overlay_labeled.png", optimize=True)
    after.append((scene_dir / "overlay_labeled.png").stat().st_size)

    b = sum(before) / 1e6 if before else 0
    a = sum(after) / 1e6
    print(f"{scene_dir.name}: {b:.1f} MB -> {a:.2f} MB")
    return b, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(DEFAULT_SCENES))
    ap.add_argument("--scene", default=None)
    ap.add_argument("--keep", action="store_true", help="rename the old files to .bak instead of overwriting")
    args = ap.parse_args()

    root = Path(args.scenes)
    dirs = [root / args.scene] if args.scene else sorted(d for d in root.iterdir() if d.is_dir())
    totals = [0.0, 0.0]
    for d in dirs:
        r = regen_one(d, keep=args.keep)
        if r:
            totals[0] += r[0]; totals[1] += r[1]
    if totals[0]:
        print(f"\ntotal: {totals[0]:.0f} MB -> {totals[1]:.1f} MB")


if __name__ == "__main__":
    main()
