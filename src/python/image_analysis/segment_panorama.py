#!/usr/bin/env python3
"""
segment_panorama.py — turn one 360° equirectangular photo into a SoundScapes scene.

Run ONCE per PoI, offline. Output is the manifest that soundscape-engine.js consumes:
per sound region an azimuth, elevation, angular spread and distance. The browser never
segments anything — at runtime "per-viewport analysis" is a frustum test against those
angles, which costs a few hundred float operations.

    pip install torch transformers pillow numpy scipy opencv-python
    python segment_panorama.py panorama.jpg --out audio/kaprun-wasserfallboden

Why not run the model on the equirectangular image directly? Because segmentation models
are trained on ordinary perspective photos, and ERP stretches everything near the poles —
a patch of sky at the top of the frame becomes a 2048-pixel-wide smear. So we reproject
into overlapping normal-looking perspective views, segment those, and accumulate the class
probabilities back onto the sphere. Overlap is what removes the seams.

Modes:
  --mode multiview  20 overlapping 90° views (default). Best masks, ~20 inferences.
  --mode cube       6 cube faces. Fast, visible seams at the edges.
  --mode erp        one pass on the raw equirect. Fastest, poles are mush. Fine if you
                    only care about the horizon band, which for a hackathon you might.

Open vocabulary: ADE20K gives you sky, tree, mountain, rock, water, waterfall, river,
lake, grass, field, path, earth, house, animal for free. It has no snow, glacier, scree,
cattle or cable car. Pass --open-vocab to add those with CLIPSeg text prompts.
"""

import argparse, json, math, re, sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# The alpine label set. Left: our sound classes. Right: the ADE20K-150 indices
# that map onto them (verified against CSAILVision/sceneparsing objectInfo150.csv).
# ---------------------------------------------------------------------------
ALPINE_CLASSES = {
    # our class     ADE20K indices (1-based as in objectInfo150.csv)
    "sky":        [3],
    "forest":     [5, 18, 73],       # tree, plant, palm
    "pasture":    [10, 30, 67],      # grass, field, flower
    "rock":       [17, 35, 69],      # mountain, rock, hill
    "scree":      [14, 47, 92, 95],  # earth, sand, dirt track, land
    "water":      [22, 27, 129, 61], # water, sea, lake, river
    "waterfall":  [114],
    "trail":      [53],
    "built":      [26, 33, 62, 49],  # house, fence, bridge, skyscraper
    "animal":     [127],             # wildlife only
    "person":     [13],              # hikers. Split out of `animal` deliberately: a
                                     # walker is not a marmot, and this is the class the
                                     # human-pressure control exists to notice.
}

# Classes CLIPSeg adds by text prompt, because ADE20K has no label for them.
#
# "water" here is deliberately NOT a new class: it's the same key as the ADE-derived
# bucket in ALPINE_CLASSES above, so its score reinforces "water" instead of competing
# with it (same trick already used for "scree" and "person" -- see the class_order /
# probs merge logic in main()). It's the cheap, targeted fix for alpine lakes: ADE20K
# does have a `lake` index (129, already mapped into "water"), so this isn't filling a
# gap in the label set the way snow/glacier/cattle are -- it's compensating for the main
# model under-calling it on a flat, often mirror-still or ice-grey lake surface, which is
# a real failure mode independent of which ADE model (SegFormer or Mask2Former) is doing
# the main pass. Needs --open-vocab; doesn't need Mask2Former or a resegmentation of
# everything to try -- CLIPSeg is a separate, much smaller/faster pass than either.
OPEN_VOCAB_PROMPTS = {
    "snow":     "a snowfield on a mountain",
    "glacier":  "a glacier of blue ice",
    "scree":    "a slope of loose grey scree and broken rock",
    "cattle":   "cows grazing on an alpine pasture",
    "cablecar": "a cable car line or ski lift pylon",
    "person":   "people hiking, walkers with backpacks",
    "water":    "a calm alpine lake or mountain tarn reflecting the sky",
}

# Sound design intent per class: base level in dB, focus bonus, default spread cap,
# and how far away it is assumed to be when geometry can't tell us.
SOUND_SPEC = {
    #            type      gain  focus  far_m   min_solid_angle_sr
    "waterfall": ("region",   2,    10,    120,  0.02),
    "water":     ("region",  -4,     9,    200,  0.05),
    "forest":    ("region",  -6,     8,    180,  0.08),
    "pasture":   ("region", -12,     7,     60,  0.08),
    "rock":      ("region", -14,     6,    400,  0.10),
    "scree":     ("region", -12,     7,    250,  0.06),
    "snow":      ("region", -16,     6,    300,  0.06),
    "glacier":   ("region", -14,     7,    500,  0.06),
    "built":     ("region", -18,     8,    150,  0.02),
    "trail":     (None,       0,     0,      0,  1.00),  # silent, informational
    "sky":       (None,       0,     0,      0,  1.00),  # silent; wind bed covers it
    "cattle":    ("event",    4,     6,    300,  0.01),
    "person":    ("event",   -4,     7,     60,  0.002),
    "animal":    ("event",    4,     6,    300,  0.01),
    "cablecar":  ("event",   -4,     8,    250,  0.005),
}

# Freesound queries, so the output manifest can be filled by fetch-sounds.mjs.
QUERIES = {
    "waterfall": "waterfall close", "water": "mountain stream brook",
    "forest": "forest wind trees ambience", "pasture": "meadow grasshopper insects summer",
    "rock": "wind gusts exposed ridge", "scree": "wind over gravel scree",
    "snow": "wind over snow", "glacier": "ice creaking glacier",
    "built": "wooden hut creak", "cattle": "cow bell alps",
    "animal": "small bird chirp single", "cablecar": "cable car motor hum",
    "person": "distant hikers voices outdoor",
}

CAMERA_HEIGHT_M = 1.6  # eye height; used for the ground-plane distance estimate
MAX_CANVAS_W = 4096    # cap on the working equirect canvas, to keep memory sane
# Hard cap on the resolution we run inference and accumulate at. The segmentation model
# sees 512x512 regardless, and the sphere grid is 512x256, so anything beyond this is
# memory spent for no accuracy. Peak cost is roughly K x side^2 x 4 bytes per view.
ANALYSIS_MAX_SIDE = 1024


def downsample_for_analysis(crop, lon, lat, max_side):
    """Shrink a view and its direction grids together, so nothing downstream desyncs."""
    import cv2
    h, w = crop.shape[:2]
    if max(h, w) <= max_side:
        return crop, lon, lat
    scale = max_side / max(h, w)
    nh, nw = max(int(h * scale), 16), max(int(w * scale), 16)
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    ys = np.linspace(0, h - 1, nh).astype(int)
    xs = np.linspace(0, w - 1, nw).astype(int)
    return crop, lon[np.ix_(ys, xs)], lat[np.ix_(ys, xs)]


def embed_partial(img, hfov, vfov):
    """
    Lift a partial panorama (iPhone sweep, or a plain photo) onto a full equirectangular
    canvas, centred, with a validity mask marking where there is actually image. Every
    step downstream then works unchanged, and regions can only form where we can see.
    """
    import cv2
    h, w = img.shape[:2]
    W = int(round(w * 360.0 / hfov))
    if W > MAX_CANVAS_W:                       # downscale the source to fit the canvas
        k = MAX_CANVAS_W / W
        img = cv2.resize(img, (max(int(w * k), 16), max(int(h * k), 16)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        W = MAX_CANVAS_W
    W += W % 2
    H = W // 2
    h_erp = max(int(round(H * vfov / 180.0)), 8)
    strip = cv2.resize(img, (w, h_erp), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((H, W, 4), np.uint8)
    x0, y0 = (W - w) // 2, (H - h_erp) // 2
    canvas[y0:y0 + h_erp, x0:x0 + w, :3] = strip
    canvas[y0:y0 + h_erp, x0:x0 + w, 3] = 255
    return canvas


# ---------------------------------------------------------------------------
# Equirectangular <-> perspective
# ---------------------------------------------------------------------------
def view_list(mode):
    """(yaw, pitch, fov) in degrees for each perspective view to segment."""
    if mode == "cube":
        return [(y, 0, 90) for y in (0, 90, 180, 270)] + [(0, 90, 90), (0, -90, 90)]
    if mode == "multiview":
        views = []
        for pitch in (-35, 0, 35):
            for yaw in range(0, 360, 60):
                views.append((yaw + (30 if pitch else 0), pitch, 95))
        views += [(0, 85, 95), (0, -85, 95)]
        return views
    raise ValueError(mode)


def render_perspective(erp, yaw, pitch, fov, size=512):
    """
    Sample a normal-looking perspective image out of the equirect panorama, and return
    it together with the ERP pixel coordinates each output pixel came from (so we can
    scatter results back onto the sphere afterwards).
    """
    import cv2
    H, W = erp.shape[:2]
    f = (size / 2) / math.tan(math.radians(fov) / 2)

    j, i = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x = (j - size / 2) / f
    y = -(i - size / 2) / f
    z = np.ones_like(x)
    v = np.stack([x, y, z], -1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)

    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    d = v @ Rx.T @ Ry.T

    lon = np.arctan2(d[..., 0], d[..., 2])          # 0 = north, + = east
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))

    u = (lon / (2 * math.pi) + 0.5) * (W - 1)
    vv = (0.5 - lat / math.pi) * (H - 1)
    crop = cv2.remap(erp, u.astype(np.float32), vv.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return crop, lon, lat


def flat_view(img, hfov, vfov):
    """
    Directions for every pixel of an ordinary photo, straight from the pinhole model.

    A flat photo needs no reprojection at all — it is already a perspective view. Lifting
    it onto an equirect canvas and then rendering perspective views back out of that
    resamples it twice and throws away detail for no gain.
    """
    h, w = img.shape[:2]
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    x = (j - w / 2) / f
    y = -(i - h / 2) / f
    z = np.ones_like(x)
    n = np.sqrt(x * x + y * y + z * z)
    lon = np.arctan2(x / n, z / n)
    lat = np.arcsin(np.clip(y / n, -1, 1))
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img, lon, lat


def scatter_to_sphere(acc, weight, vacc, probs, valid, lon, lat):
    """Accumulate per-class probabilities and image validity onto the sphere grid."""
    K, GH, GW = acc.shape
    gx = np.clip(((lon / (2 * math.pi) + 0.5) * GW).astype(int), 0, GW - 1)
    gy = np.clip(((0.5 - lat / math.pi) * GH).astype(int), 0, GH - 1)
    flat = (gy * GW + gx).ravel()
    for k in range(K):
        np.add.at(acc[k].ravel(), flat, probs[k].ravel())
    np.add.at(vacc.ravel(), flat, valid.ravel().astype(np.float32))
    np.add.at(weight.ravel(), flat, np.ones_like(flat, dtype=np.float32))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def image_processor(cls, model_id):
    """
    Recent transformers defaults to a FAST image processor implemented with torchvision,
    and raises ImportError if torchvision is absent. The original numpy/PIL processor does
    the same job here (we feed it one image at a time), so fall back to it rather than
    making torchvision a hard requirement.
    """
    try:
        return cls.from_pretrained(model_id)
    except ImportError:
        try:
            return cls.from_pretrained(model_id, use_fast=False)
        except Exception:
            raise SystemExit(
                "This model's image processor needs torchvision, and the slow fallback "
                "was refused too.\n  pip install torchvision")


def load_ade_model(model_id, device):
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    proc = image_processor(AutoImageProcessor, model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return proc, model


def load_segmentation_model(model_id, device):
    """
    Dispatch on model architecture. SegFormer is per-pixel classification (one softmax
    over 150 classes at every location), so `ade_probs` can read its logits directly.
    Mask2Former is mask-classification instead: a fixed set of query masks, each with its
    OWN class distribution, no per-pixel logits at all. Loading a Mask2Former checkpoint
    through SegformerForSemanticSegmentation.from_pretrained() does not degrade gracefully
    -- it raises immediately, which is why swapping in --model
    facebook/mask2former-swin-large-ade-semantic never actually worked before. Route each
    architecture to its own loader and its own probs function instead.
    """
    if "mask2former" in model_id.lower():
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        proc = image_processor(AutoImageProcessor, model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id).to(device).eval()
        return proc, model, mask2former_probs
    proc, model = load_ade_model(model_id, device)
    return proc, model, ade_probs


def ade_probs(proc, model, device, crop, class_order):
    """
    Run the ADE20K model and collapse its 150 channels onto our ~10 alpine classes.

    ORDER MATTERS, and getting it wrong will take the machine down. The model emits
    150 channels at about 128x128. Upsampling those to the input resolution FIRST and
    collapsing afterwards costs 150 x H x W floats — for a 6144x3072 equirect that is
    11 GB per tensor, three times over (interpolate, softmax, copy to host). On unified
    memory that is not an OOM exception, it is a frozen machine.

    So: softmax and collapse at the model's own resolution (150 x 128 x 128 is 10 MB),
    then interpolate only the ~10 channels we actually keep.
    """
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        inputs = proc(images=Image.fromarray(crop), return_tensors="pt").to(device)
        logits = model(**inputs).logits                  # 1 x 150 x h x w, h,w ~ 128
        p = logits.softmax(1)                            # still small

        K = len(class_order)
        small = torch.zeros((1, K) + tuple(p.shape[2:]), device=p.device, dtype=p.dtype)
        for n, name in enumerate(class_order):
            for idx in ALPINE_CLASSES.get(name, []):
                small[0, n] += p[0, idx - 1]             # objectInfo150.csv is 1-based

        out = F.interpolate(small, size=crop.shape[:2], mode="bilinear",
                            align_corners=False)
        return out[0].float().cpu().numpy()


def mask2former_probs(proc, model, device, crop, class_order):
    """
    Run Mask2Former and collapse its ADE label space onto our alpine classes, the same way
    ade_probs does for SegFormer -- but Mask2Former has no per-pixel logits to begin with.
    It predicts a fixed number of query masks (num_queries, ~100-200), each carrying its
    own softmax over the 150 ADE classes plus "no object". Turning that into a per-pixel,
    per-class probability map is exactly the first half of what transformers' own
    post_process_semantic_segmentation does internally (softmax the class logits, sigmoid
    the mask logits, einsum them together) -- we stop one step short of its final argmax so
    the full per-class map can be collapsed and accumulated onto the sphere exactly like
    the SegFormer path.

    Same memory discipline as ade_probs: the einsum and the 150->K collapse happen at the
    model's own small mask resolution, and only the ~K channels we keep get interpolated up
    to the crop size.
    """
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        inputs = proc(images=Image.fromarray(crop), return_tensors="pt").to(device)
        outputs = model(**inputs)
        class_logits = outputs.class_queries_logits           # 1 x Q x (150 + "no object")
        mask_logits = outputs.masks_queries_logits             # 1 x Q x h' x w' (small)

        class_probs = class_logits.softmax(-1)[..., :-1]       # drop "no object"
        mask_probs = mask_logits.sigmoid()
        per_class = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)  # 1x150xh'xw'

        K = len(class_order)
        small = torch.zeros((1, K) + tuple(per_class.shape[2:]), device=per_class.device,
                            dtype=per_class.dtype)
        for n, name in enumerate(class_order):
            for idx in ALPINE_CLASSES.get(name, []):
                small[0, n] += per_class[0, idx - 1]           # objectInfo150.csv is 1-based

        out = F.interpolate(small, size=crop.shape[:2], mode="bilinear",
                            align_corners=False)
        return out[0].float().cpu().numpy()


def clipseg_probs(crop, prompts, device):
    """Open-vocabulary pass for the classes ADE20K has no label for."""
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    global _CLIPSEG
    if "_CLIPSEG" not in globals() or _CLIPSEG is None:
        proc = image_processor(CLIPSegProcessor, "CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined").to(device).eval()
        _CLIPSEG = (proc, model)
    proc, model = _CLIPSEG
    img = Image.fromarray(crop)
    texts = list(prompts.values())
    with torch.no_grad():
        inputs = proc(text=texts, images=[img] * len(texts),
                      padding=True, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        if logits.ndim == 2:
            logits = logits[None]
        m = torch.sigmoid(logits).cpu().numpy()
    out = np.stack([np.array(Image.fromarray(mk).resize(
        (crop.shape[1], crop.shape[0]), Image.BILINEAR)) for mk in m])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Sphere label map -> sound regions
# ---------------------------------------------------------------------------
def regions_from_labels(labels, class_order, grid_h, grid_w, max_per_class=3):
    """
    Connected components per class, then for each blob: solid angle, centroid direction,
    angular spread. cos(latitude) weighting matters — without it the poles, which are a
    handful of real steradians, dominate every average.
    """
    from scipy import ndimage
    lat = (0.5 - (np.arange(grid_h) + 0.5) / grid_h) * math.pi
    dlat = math.pi / grid_h
    dlon = 2 * math.pi / grid_w
    cell_sr = (np.cos(lat) * dlat * dlon)[:, None] * np.ones((1, grid_w))

    out = []
    for n, name in enumerate(class_order):
        spec = SOUND_SPEC.get(name)
        if not spec or spec[0] is None:
            continue
        ltype, gain, focus, far_m, min_sr = spec
        found = []          # components for this class, before merging or capping

        mask = labels == n
        if not mask.any():
            continue
        comp, ncomp = ndimage.label(mask, structure=np.ones((3, 3)))

        # ndimage doesn't know the panorama wraps, so a region sitting on the seam
        # comes back as two half-loud sources on opposite sides. Union them.
        parent = list(range(ncomp + 1))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for y in range(grid_h):
            for dy in (-1, 0, 1):
                yy = y + dy
                if 0 <= yy < grid_h and comp[y, 0] and comp[yy, grid_w - 1]:
                    union(comp[y, 0], comp[yy, grid_w - 1])
        for c in range(1, ncomp + 1):
            comp[comp == c] = find(c)

        for c in sorted(set(comp[comp > 0].tolist())):
            m = comp == c
            sr = float(cell_sr[m].sum())
            if sr < min_sr:
                continue

            ys, xs = np.nonzero(m)
            w = cell_sr[ys, xs]
            # Column 0 is longitude -pi. The panorama's CENTRE column is treated as
            # north; set north_offset_deg in the manifest to correct to true north.
            # Azimuth must be averaged as a vector, or a blob straddling the seam
            # averages to the opposite side of the sphere.
            az = ((xs + 0.5) / grid_w - 0.5) * 2 * math.pi
            el = lat[ys]
            cx = float((np.cos(az) * w).sum()); sx = float((np.sin(az) * w).sum())
            az_c = math.degrees(math.atan2(sx, cx)) % 360
            el_c = math.degrees(float((el * w).sum() / w.sum()))

            # Angular radius of a cap with this solid angle: sr = 2*pi*(1-cos(r))
            spread = math.degrees(math.acos(max(-1.0, 1.0 - sr / (2 * math.pi))))
            spread = float(np.clip(spread, 12, 100))

            # Distance: put anything below the horizon on the ground plane. Use the
            # region's FAR edge (its shallowest depression angle), not its centroid —
            # the centroid of a big pasture sits at your feet, but what you hear as
            # "the pasture over there" is its far side. Being 30 % wrong is inaudible;
            # being on the wrong side of the listener is very audible.
            el_far = float(np.percentile(np.degrees(el), 85))  # closest to the horizon
            if el_far < -1.5:
                dist = CAMERA_HEIGHT_M / math.tan(math.radians(-el_far))
                dist = float(np.clip(dist, 5, far_m * 2))
            else:
                dist = float(far_m)

            found.append({
                "class": name, "type": ltype,
                "az": round(az_c, 1), "el": round(el_c, 1),
                "spread": round(spread, 1), "distance": round(dist),
                "gain": gain, "focus": focus,
                "solid_angle_sr": round(sr, 4),
                "query": QUERIES.get(name, name),
                "_vec": (cx, sx), "_w": float(w.sum()),
            })

        if not found:
            continue
        found.sort(key=lambda r: -r["solid_angle_sr"])

        if ltype == "event":
            # One layer per class, not one per individual. Twenty hikers are not twenty
            # sound sources needing twenty files — they are one event layer whose
            # scheduler picks a direction inside their combined extent each time it
            # fires. Merging also stops a crowd from being twenty times as loud.
            cx = sum(r["_vec"][0] for r in found)
            sx = sum(r["_vec"][1] for r in found)
            total_sr = sum(r["solid_angle_sr"] for r in found)
            az_c = math.degrees(math.atan2(sx, cx)) % 360
            el_c = sum(r["el"] * r["solid_angle_sr"] for r in found) / max(total_sr, 1e-9)
            # Spread must cover where they actually are, not just their combined area.
            spans = [abs(((r["az"] - az_c + 180) % 360) - 180) for r in found]
            spread = float(np.clip(max(spans + [8.0]) + 6.0, 12, 100))
            dist = sorted(r["distance"] for r in found)[len(found) // 2]
            merged = dict(found[0])
            merged.update(id=name, az=round(az_c, 1), el=round(el_c, 1),
                          spread=round(spread, 1), distance=dist,
                          solid_angle_sr=round(total_sr, 4), count=len(found))
            out.append(merged)
        else:
            # Regions: a big face split by an occluder genuinely wants two directions,
            # but nine of them is a manifest nobody will tune. Keep the largest few.
            for i, r in enumerate(found[:max_per_class], 1):
                r = dict(r)
                r["id"] = name if len(found[:max_per_class]) == 1 else f"{name}-{i}"
                out.append(r)

    for r in out:
        r.pop("_vec", None)
        r.pop("_w", None)
    out.sort(key=lambda r: -r["solid_angle_sr"])
    return out


def class_shares(acc, valid, class_order):
    """
    How much of what the camera saw is each class, as fractions summing to 1.

    Weighted by SOLID ANGLE, not pixel count: an equirectangular frame devotes as many
    pixels to the zenith as to the horizon, so a pixel count would report a scene that is
    mostly sky no matter what was in front of you. cos(latitude) fixes that.

    Only the photographed part of the sphere counts — for a 150-degree sweep the other
    210 degrees are not "other", they are simply not in the picture.

    "other" is principled rather than a threshold: the ADE model distributes probability
    over 150 classes and we map about ten of them, so whatever mass lands on the 140 we
    do not model is, precisely, "the model saw something we have no category for".
    """
    K, GH, GW = acc.shape
    lat = (0.5 - (np.arange(GH) + 0.5) / GH) * math.pi
    cell = (np.cos(lat) * (math.pi / GH) * (2 * math.pi / GW))[:, None] * np.ones((1, GW))
    cell = cell * (valid >= 0.5)
    if cell.sum() <= 0:
        return []

    mapped = np.clip(acc.sum(0), 0.0, None)
    other = np.clip(1.0 - mapped, 0.0, 1.0)          # unmodelled ADE classes
    stack = np.concatenate([acc, other[None]], 0)
    stack = stack / np.maximum(stack.sum(0, keepdims=True), 1e-9)

    shares = (stack * cell[None]).sum((1, 2))
    shares = shares / max(shares.sum(), 1e-9)
    return list(zip(list(class_order) + ["other"], shares.tolist()))


def as_percentages(shares, floor=1):
    """
    Integer percentages that actually sum to 100, by largest remainder. Classes below the
    floor are dropped and their share folded into "other", because a 0 % row is noise.
    """
    keep = [(n, v) for n, v in shares if v * 100 >= floor or n == "other"]
    dropped = sum(v for n, v in shares if (n, v) not in keep)
    keep = [(n, v + (dropped if n == "other" else 0.0)) for n, v in keep]
    total = sum(v for _, v in keep) or 1.0

    scaled = [(n, v / total * 100) for n, v in keep]
    ints = [(n, int(math.floor(x))) for n, x in scaled]
    remainder = 100 - sum(i for _, i in ints)
    order = sorted(range(len(scaled)), key=lambda k: -(scaled[k][1] - ints[k][1]))
    ints = [list(t) for t in ints]
    for k in order[:max(remainder, 0)]:
        ints[k][1] += 1

    rows = [(n, v) for n, v in ints if v > 0]
    # Biggest first, but "other" always last — it is a residual, not a finding.
    rows.sort(key=lambda t: (t[0] == "other", -t[1]))
    return rows


def class_report_name(source_name):
    """<original image filename without extension>_classes.txt"""
    return f"{Path(source_name).stem}_classes.txt"


def write_class_report(rows, out_dir, panorama_file, mode, source_name=None):
    path = out_dir / class_report_name(source_name or panorama_file)
    width = max((len(n) for n, _ in rows), default=8)
    lines = [
        f"# class coverage of {panorama_file} ({mode} pass)",
        "# solid-angle weighted over the photographed area, so these are shares of what",
        "# the camera saw — not pixel counts, and not shares of the whole sphere.",
        "# 'other' is probability the model gave to classes we do not map.",
        "",
    ]
    lines += [f"{n:<{width}} {v}%" for n, v in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def sky_sanity(cells, gh, gw, names, hfov):
    """
    On a full sphere, roughly half of what you see is above the horizon and most of that
    is sky. If the upper hemisphere comes back with almost no sky, the segmentation is
    wrong — not slightly, but in a way that will place water sources overhead.

    Returns a warning string, or None.
    """
    import numpy as np, math
    if hfov < 350 or "sky" not in names:
        return None                       # only meaningful for a full 360
    sky_i = names.index("sky")
    lat = (0.5 - (np.arange(gh) + 0.5) / gh) * math.pi
    cell = (np.cos(lat) * (math.pi / gh) * (2 * math.pi / gw))[:, None] * np.ones((1, gw))
    up = lat > 0
    total = cell[up].sum()
    if total <= 0:
        return None
    sky = cell[up][cells[up] == sky_i].sum()
    share = 100 * sky / total
    if share >= 20:
        return None
    worst = {}
    for i, n in enumerate(names):
        w = cell[up][cells[up] == i].sum()
        if w > 0:
            worst[n] = 100 * w / total
    top = sorted(worst.items(), key=lambda t: -t[1])[:2]
    return (f"only {share:.0f}% of the sky is labelled 'sky' — mostly "
            + ", ".join(f"{n} {v:.0f}%" for n, v in top)
            + ".\n  A full 360 is about half sky. This segmentation is wrong. Re-run with"
              "\n  --mode multiview and Mask2Former-Swin-L (the default), not smoke-test"
              "\n  settings: a small model on a raw equirect reads cloud as rock or water.")


PALETTE = {
    "sky": (110, 165, 200), "forest": (48, 92, 58), "pasture": (140, 176, 92),
    "rock": (140, 134, 126), "scree": (176, 168, 150), "water": (60, 130, 168),
    "waterfall": (200, 232, 240), "trail": (196, 160, 110), "built": (188, 96, 70),
    "animal": (226, 146, 87), "snow": (240, 244, 248), "glacier": (150, 205, 220),
    "cattle": (214, 122, 70), "cablecar": (120, 100, 140),
    "person": (232, 72, 96),
}


def _load_font(size):
    """A real font if the OS has one, PIL's bitmap font otherwise."""
    from PIL import ImageFont
    for cand in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def direction_to_pixel(az, el, w, h, projection, hfov, vfov):
    """
    Where a direction lands in the SOURCE image. Returns (x, y) or None if it is behind
    the camera. Each projection maps differently, and azimuth 0 is the image centre.
    """
    lon = ((az + 180.0) % 360.0) - 180.0
    if projection == "equirect":
        return (lon / 360.0 + 0.5) * w, (0.5 - el / 180.0) * h
    if projection == "cylindrical":
        return (lon / hfov + 0.5) * w, (0.5 - el / vfov) * h
    # flat: gnomonic, the inverse of flat_view's pinhole model
    lo, la = math.radians(lon), math.radians(el)
    X = math.cos(la) * math.sin(lo)
    Y = math.sin(la)
    Z = math.cos(la) * math.cos(lo)
    if Z <= 1e-6:
        return None
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    return w / 2 + f * (X / Z), h / 2 - f * (Y / Z)


def draw_region_labels(img, regions, shares, projection, hfov, vfov, max_labels=12,
                       size_px=None):
    """
    Draw the class-name / coverage% / distance callout for each region onto `img` (an
    RGBA PIL Image, modified in place and returned) at its projected (az, el) position.

    Shared by write_labeled_overlay() below (the pitch asset: labels baked onto the
    photo's segmentation tint) and by regen_overlays.py (which rebuilds both that and a
    labels-only transparent version straight from scene.segmented.json, no model involved)
    — sizing and layout only ever need to change in one place.

    `regions` may be either the full in-process region dicts (which carry a `class` key)
    or the reduced layer dicts persisted into scene.segmented.json (which don't -- the
    class name there is recovered from the `id`, e.g. "forest-2" -> "forest").

    `size_px` overrides the automatic width-relative sizing, for callers that want a fixed
    size regardless of the source photo's resolution.
    """
    from PIL import ImageDraw
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    # 65% of the 13 Sep sizing (max(10, w/130)) -- still too big/clumsy at that size per
    # Tom, so floor and width-divisor are both scaled by the same 0.65. Padding, corner
    # radius, outline width and the centroid dot below are all defined as fractions of
    # `size`, so they shrink proportionally for free.
    size = size_px or max(7, int(w / 200))
    font = _load_font(size)
    pct = shares if isinstance(shares, dict) else dict(shares)

    for r in sorted(regions, key=lambda r: -r.get("solid_angle_sr", 0))[:max_labels]:
        cls = r.get("class") or re.sub(r"-\d+$", "", r["id"])
        pos = direction_to_pixel(r["az"], r["el"], w, h, projection, hfov, vfov)
        if pos is None:
            continue
        x, y = pos
        if not (-w < x < 2 * w and 0 <= y <= h):
            continue
        x = min(max(x, size), w - size)
        y = min(max(y, size), h - size)

        colour = PALETTE.get(cls, (200, 200, 200))
        share = pct.get(cls)
        text = r["id"] if share is None else f"{r['id']}  {share}%"
        text += f"\n{r['distance']} m"

        box = draw.multiline_textbbox((x, y), text, font=font, anchor="mm", spacing=1)
        pad = size * 0.28
        # Fill alpha dropped 205 -> 120 (~47% opacity, was ~80%) so the photo reads
        # through the box instead of the box reading as an opaque plate on the image.
        draw.rounded_rectangle(
            [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
            radius=size * 0.26, fill=(8, 12, 11, 120), outline=colour + (255,),
            width=max(1, size // 14))
        draw.multiline_text((x, y), text, font=font, fill=(240, 246, 242, 255),
                            anchor="mm", align="center", spacing=1)
        # A dot marks the exact centroid, since the label is nudged to stay on canvas.
        rr = max(2, size // 8)
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=colour + (255,))
    return img


def write_labeled_overlay(base, regions, shares, out_dir, projection, hfov, vfov,
                          max_labels=12):
    """
    Two outputs, same labels, different backgrounds:

    - overlay_labeled.png: the labels baked onto the same transparent tint as overlay.png
      (see write_pngs). The pitch asset — one picture that makes the pipeline legible to
      someone who has never seen the code, proving 'waterfall' really is on a waterfall.
    - overlay_labels.png: the SAME labels, on a fully transparent background with no tint
      at all. This is what the viewer's "Labels" toggle actually loads, so switching
      labels on/off no longer means the browser has to reverse-engineer which pixels are
      "label" by diffing two baked-together images.

    `base` is the RGBA tint `write_pngs` returned — kept as RGBA here (not flattened to
    RGB) so the text and dots land on the same mostly-transparent image, not a copy of
    the photo.
    """
    img = Image.fromarray(base, mode="RGBA") if base.shape[-1] == 4 else Image.fromarray(base).convert("RGBA")
    labels_only = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_region_labels(img, regions, shares, projection, hfov, vfov, max_labels)
    draw_region_labels(labels_only, regions, shares, projection, hfov, vfov, max_labels)

    dest = out_dir / "overlay_labeled.png"
    img.save(dest, optimize=True)
    labels_dest = out_dir / "overlay_labels.png"
    labels_only.save(labels_dest, optimize=True)
    return dest, labels_dest


TINT_ALPHA = 132         # opacity of overlay.png / overlay_labeled.png over the photo
VOID_RGB = (10, 12, 12)  # lut[255]: outside the photographed area


def write_pngs(labels, class_order, erp, out_dir, grid_w, grid_h, lonlat=None):
    """
    labels.png is the sphere label map. overlay.png is a TRANSPARENT TINT of it, sized to
    the photo, meant to be drawn on top of the photo in the browser — not the photo with
    the labels blended in.

    Baking the label colours into the photograph (as this used to do: 0.62*photo +
    0.38*colour, saved as a lossless PNG) bakes the photo's own high-frequency detail
    into the file too, which is exactly what a PNG compresses worst — that came to
    20+ MB for one panorama. A flat-coloured image that is mostly transparent (opaque
    only over classified ground) is exactly what PNG compresses BEST, and the browser
    already has the photo to layer it over, so nothing is lost by leaving it out.

    For an equirect canvas the label map can simply be stretched to the photo's size. For
    a flat photo it cannot: the photo is not 2:1 and its pixels are not linear in
    longitude. So when `lonlat` is supplied we look each base pixel's own direction up in
    the map instead.
    """
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(class_order):
        lut[i] = PALETTE.get(c, (128, 128, 128))
    lut[255] = VOID_RGB
    lab_rgb = lut[labels]
    Image.fromarray(lab_rgb).resize((grid_w * 2, grid_h * 2), Image.NEAREST) \
        .save(out_dir / "labels.png")

    h, w = erp.shape[:2]
    if lonlat is None:
        big = np.array(Image.fromarray(lab_rgb).resize((w, h), Image.NEAREST))
    else:
        lon, lat = lonlat
        GH, GW = labels.shape
        gx = np.clip(((lon / (2 * math.pi) + 0.5) * GW).astype(int), 0, GW - 1)
        gy = np.clip(((0.5 - lat / math.pi) * GH).astype(int), 0, GH - 1)
        big = lab_rgb[gy, gx]

    void = np.all(big == VOID_RGB, axis=-1)
    alpha = np.where(void, 0, TINT_ALPHA).astype(np.uint8)
    tint = np.dstack([big, alpha])
    Image.fromarray(tint, mode="RGBA").save(out_dir / "overlay.png", optimize=True)
    return tint


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panorama")
    ap.add_argument("--out", default="out")
    ap.add_argument("--mode", default="multiview", choices=["multiview", "cube", "erp"])
    ap.add_argument("--model", default="facebook/mask2former-swin-large-ade-semantic",
                    help="ADE-trained model to segment with. Defaults to the quality "
                         "option (Mask2Former-Swin-L, ~4x slower than SegFormer) because "
                         "this runs once per PoI, offline -- speed doesn't matter (see "
                         "segmentation-approach.md). Pass "
                         "nvidia/segformer-b4-finetuned-ade-512-512 for a fast smoke test.")
    ap.add_argument("--open-vocab", action="store_true",
                    help="add snow / glacier / scree / cattle / cable car via CLIPSeg")
    ap.add_argument("--grid", type=int, default=64,
                    help="target cells across the PHOTOGRAPHED width (default %(default)s). "
                         "Scaled up automatically for a narrow field of view.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--projection", default=None,
                    choices=["equirect", "cylindrical", "flat"],
                    help="default: read from the scene's panorama block, else equirect")
    ap.add_argument("--hfov", type=float, default=None, help="horizontal degrees covered")
    ap.add_argument("--vfov", type=float, default=None, help="vertical degrees covered")
    ap.add_argument("--max-regions-per-class", type=int, default=3,
                    help="keep at most this many region layers per class (default "
                         "%(default)s). Event classes are always merged into one.")
    ap.add_argument("--analysis-max-side", type=int, default=ANALYSIS_MAX_SIDE,
                    help="cap on the resolution inference runs at (default %(default)s). "
                         "Raising this buys no accuracy and costs memory quadratically.")
    ap.add_argument("--scene-json", default=None,
                    help="read projection/hfov/vfov from this scene.json (written by "
                         "ingest_images.py) instead of passing them by hand")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if
             getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
             else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    src_img = np.array(Image.open(args.panorama).convert("RGB"))

    projection, hfov, vfov = args.projection, args.hfov, args.vfov
    source_name = Path(args.panorama).name
    if args.scene_json:
        pano = json.loads(Path(args.scene_json).read_text()).get("panorama", {})
        projection = projection or pano.get("projection")
        hfov = hfov if hfov is not None else pano.get("hfov_deg")
        vfov = vfov if vfov is not None else pano.get("vfov_deg")
        # ingest_images.py renames every image to panorama.jpg but records where it
        # came from, so the report can carry the name you actually recognise.
        source_name = pano.get("source", source_name)
    if projection is None:
        h0, w0 = src_img.shape[:2]
        projection = "equirect" if abs(w0 / h0 - 2) < 0.06 else "cylindrical"
        print(f"no projection given; guessing {projection} from {w0}x{h0}")
    if hfov is None:
        hfov = 360.0 if projection == "equirect" else 150.0
    if vfov is None:
        h0, w0 = src_img.shape[:2]
        vfov = 180.0 if projection == "equirect" else hfov * h0 / w0

    flat = projection == "flat"
    if flat:
        # No canvas: we segment the photo itself and compute directions per pixel.
        erp = np.dstack([src_img, np.full(src_img.shape[:2], 255, np.uint8)])
        print(f"flat photo {hfov:.0f}x{vfov:.0f}deg — segmented directly, no reprojection")
    elif projection == "equirect" and hfov > 350:
        erp = np.dstack([src_img, np.full(src_img.shape[:2], 255, np.uint8)])
    else:
        # A sweep panorama or a plain photo only covers part of the sphere. Lift it onto
        # a full canvas with a validity mask so nothing downstream has to special-case it.
        erp = embed_partial(src_img, hfov, vfov)
        print(f"{projection} {hfov:.0f}x{vfov:.0f}deg embedded on a "
              f"{erp.shape[1]}x{erp.shape[0]} sphere canvas")
    H, W = erp.shape[:2]

    class_order = [c for c in ALPINE_CLASSES]
    if args.open_vocab:
        class_order += [c for c in OPEN_VOCAB_PROMPTS if c not in class_order]

    # Grid resolution follows the FIELD OF VIEW, not the sphere. A 69-degree photo covers
    # a fifth of the sphere, so at a fixed 64-column grid only ~12 columns land inside the
    # picture and every region comes out as one coarse blob. Scale up so the photographed
    # area always gets roughly `--grid` cells across it.
    gw = int(np.clip(round(args.grid * 360.0 / max(hfov, 30.0)), args.grid, 512))
    gw += gw % 2                     # keep the 4x downsample exact
    gh = gw // 2
    GW = gw * 4                      # label map resolution
    GH = GW // 2
    acc = np.zeros((len(class_order), GH, GW), np.float32)
    weight = np.zeros((GH, GW), np.float32)
    vacc = np.zeros((GH, GW), np.float32)

    proc, model, probs_fn = load_segmentation_model(args.model, device)
    n_ade = len(ALPINE_CLASSES)

    # Say what this will cost before allocating it. K channels x analysis area x 4 bytes,
    # plus the sphere accumulator. If this line ever prints gigabytes, stop and report it.
    side = args.analysis_max_side
    per_view_mb = len(class_order) * side * (side // 2 if args.mode == "erp" else side) * 4 / 1e6
    acc_mb = len(class_order) * GH * GW * 4 / 1e6
    print(f"working resolution <= {side} px | sound grid {gw}x{gh}, label map {GW}x{GH} | "
          f"~{per_view_mb:.0f} MB per view, {acc_mb:.0f} MB accumulator")

    views = [(0, 0, 0)] if (args.mode == "erp" or flat) else view_list(args.mode)
    # Don't spend inference on directions a partial panorama never saw.
    if hfov < 350 and args.mode != "erp" and not flat:
        views = [(y, p, f) for (y, p, f) in views
                 if abs(((y + 180) % 360) - 180) <= hfov / 2 + f / 2]
    for n, (yaw, pitch, fov) in enumerate(views, 1):
        if flat:
            crop, lon, lat = flat_view(erp, hfov, vfov)
        elif args.mode == "erp":
            crop = erp
            lon = (np.linspace(0, 1, W)[None, :] - 0.5) * 2 * math.pi * np.ones((H, 1))
            lat = (0.5 - np.linspace(0, 1, H)[:, None]) * math.pi * np.ones((1, W))
        else:
            crop, lon, lat = render_perspective(erp, yaw, pitch, fov)
        crop, lon, lat = downsample_for_analysis(crop, lon, lat, args.analysis_max_side)
        rgb = np.ascontiguousarray(crop[..., :3])
        valid = crop[..., 3].astype(np.float32) / 255.0
        if valid.mean() < 0.02:
            continue
        print(f"  [{n}/{len(views)}] yaw={yaw:>4} pitch={pitch:>3}  "
              f"{valid.mean()*100:3.0f}% image …", flush=True)

        probs = np.zeros((len(class_order),) + rgb.shape[:2], np.float32)
        probs[:n_ade] = probs_fn(proc, model, device, rgb, class_order[:n_ade])
        if args.open_vocab:
            extra = clipseg_probs(rgb, OPEN_VOCAB_PROMPTS, device)
            # CLIPSeg's sigmoid score for one prompt is an independent confidence, not a
            # class competing fairly in the same distribution as the ADE softmax (which
            # already sums to <=1 across our classes at every pixel). Added in raw, as this
            # used to do, a mediocre-but-unbounded CLIPSeg score -- and CLIPSeg is
            # genuinely bad at telling a pale, sunlit hut wall or a bright scree slope from
            # "a snowfield on a mountain" -- can outvote a real, better-calibrated ADE
            # prediction for "built" or "scree" at the very same pixel. That is what was
            # putting snow on the mountain hut and between the scree.
            # Fix: CLIPSeg only gets to claim the probability mass ADE hasn't already
            # committed elsewhere. Scale each prompt's score by the pixel's unclaimed
            # headroom (1 - the strongest ADE class there). Where ADE is confident (a clear
            # roofline, a clear rock face) CLIPSeg is squeezed out; where ADE is genuinely
            # unsure (a texture it has no label for at all, like snow) CLIPSeg decides.
            headroom = np.clip(1.0 - probs[:n_ade].max(0), 0.0, 1.0)
            for i, name in enumerate(OPEN_VOCAB_PROMPTS):
                probs[class_order.index(name)] += extra[i] * headroom
        probs *= valid[None]
        scatter_to_sphere(acc, weight, vacc, probs, valid, lon, lat)
        del probs, rgb, crop
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    acc /= np.maximum(weight, 1e-6)[None]
    valid_hi = vacc / np.maximum(weight, 1e-6)
    labels_hi = acc.argmax(0).astype(np.uint8)
    VOID = 255
    labels_hi[valid_hi < 0.5] = VOID

    # Downsample to the sound grid by majority vote, then derive regions.
    fy, fx = GH // gh, GW // gw
    votes = np.zeros((len(class_order), gh, gw), np.float32)
    for k in range(len(class_order)):
        votes[k] = (labels_hi == k).reshape(gh, fy, gw, fx).mean((1, 3))
    labels_lo = votes.argmax(0).astype(np.uint8)
    valid_lo = (labels_hi != VOID).reshape(gh, fy, gw, fx).mean((1, 3))
    labels_lo[valid_lo < 0.5] = VOID     # regions_from_labels skips anything not a class

    regions = regions_from_labels(labels_lo, class_order, gh, gw,
                                  args.max_regions_per_class)
    overlay_lonlat = None
    if flat:
        _, lon_f, lat_f = flat_view(erp, hfov, vfov)
        overlay_lonlat = (lon_f, lat_f)
    blended = write_pngs(labels_hi, class_order, erp, out_dir, GW, GH, overlay_lonlat)

    warn = sky_sanity(labels_lo, gh, gw, class_order, hfov)
    if warn:
        print(f"\n!! {warn}\n")

    shares = class_shares(acc, valid_hi, class_order)
    rows = as_percentages(shares)
    classes_path = write_class_report(rows, out_dir, Path(args.panorama).name,
                                      args.mode, source_name)

    # Carry forward everything ingest_images.py established about this place. Losing
    # north_offset_deg here would silently turn real compass bearings back into offsets
    # from the image centre.
    carried = {}
    if args.scene_json and Path(args.scene_json).exists():
        prev = json.loads(Path(args.scene_json).read_text())
        carried["north_offset_deg"] = prev.get("north_offset_deg", 0)
        carried["name"] = prev.get("name")
        for k in ("lat", "lon", "elev_m", "gps_error_m", "utc", "heading_deg",
                  "heading_ref", "source"):
            if k in prev.get("panorama", {}):
                carried.setdefault("_pano", {})[k] = prev["panorama"][k]

    manifest = {
        "id": out_dir.name,
        "name": args.name or carried.get("name") or out_dir.name.replace("-", " ").title(),
        "north_offset_deg": carried.get("north_offset_deg", 0),
        "panorama": {**carried.get("_pano", {}),
                     "file": Path(args.panorama).name, "projection": projection,
                     "hfov_deg": round(float(hfov), 1), "vfov_deg": round(float(vfov), 1),
                     "wrap": projection == "equirect" and hfov > 350},
        "source": {"mode": args.mode, "model": args.model,
                   "arch": "mask2former" if "mask2former" in args.model.lower()
                           else "segformer",
                   "open_vocab": args.open_vocab},
        "scene": {
            # Rough stand-ins. Replace with your scenicness regressor when you have one.
            "scenicness": round(float(np.clip(
                5 + 3 * sum(r["solid_angle_sr"] for r in regions
                            if r["class"] in ("water", "waterfall", "forest", "pasture"))
                / max(sum(r["solid_angle_sr"] for r in regions), 1e-6), 0, 10)), 1),
            "eventfulness": round(float(np.clip(
                3 + 8 * sum(r["solid_angle_sr"] for r in regions
                            if r["class"] in ("water", "waterfall", "animal", "cattle"))
                / max(sum(r["solid_angle_sr"] for r in regions), 1e-6), 0, 10)), 1),
            # Starting value for the human-pressure control, from what is visibly human
            # in the frame. Square-rooted on purpose: one hiker changes how a place feels
            # far more than their 1% of the pixels suggests. Detected people SET the
            # baseline rather than being hidden by it — if there are people in the
            # picture, the place already has people in it.
            "pressure": round(min(10.0, 2.0 * math.sqrt(
                dict(rows).get("person", 0) * 1.0
                + dict(rows).get("cablecar", 0) * 0.8
                + dict(rows).get("built", 0) * 0.5)), 1),
        },
        "classes": {n: v for n, v in rows},
        "grid": {"w": gw, "h": gh, "classes": class_order,
                 "cells": labels_lo.flatten().tolist()},
        "layers": [{k: v for k, v in {
                        "id": r["id"], "type": r["type"], "az": r["az"], "el": r["el"],
                        "spread": r["spread"], "distance": r["distance"],
                        "gain": r["gain"], "focus": r["focus"],
                        "count": r.get("count"), "query": r["query"],
                        "src": f"{r['class']}-1.mp3",
                    }.items() if v is not None} for r in regions],
    }
    (out_dir / "scene.segmented.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{len(regions)} sound regions")
    for r in regions[:14]:
        print(f"  {r['id']:<14} az {r['az']:>5.1f}  el {r['el']:>5.1f}  "
              f"spread {r['spread']:>4.0f}  {r['distance']:>4} m  {r['solid_angle_sr']:.3f} sr")
    human = {k: dict(rows).get(k, 0) for k in ("person", "built", "cablecar")}
    print("\nclass coverage (of what the camera saw):")
    for n, v in rows:
        print(f"  {n:<12} {v:>3}%")
    if any(human.values()):
        print(f"\nvisibly human: " + ", ".join(f"{k} {v}%" for k, v in human.items() if v)
              + f" -> starting pressure {manifest['scene']['pressure']}")
    labeled, labels_only = write_labeled_overlay(blended, regions, rows, out_dir,
                                                 projection, hfov, vfov)
    print(f"\nwrote {out_dir}/scene.segmented.json, {classes_path.name},")
    print(f"      labels.png, overlay.png, {labeled.name}, {labels_only.name}")
    print("Next: fill the `src` files (fetch-sounds.mjs uses the `query` fields),")
    print("then load scene.segmented.json in demo.html.")


if __name__ == "__main__":
    sys.exit(main())
