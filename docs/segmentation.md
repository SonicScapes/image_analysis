# Image analysis

## Nothing segments at runtime

A photo is static. Segmenting it per viewport recomputes an answer that never changes, at
30 fps, on a phone. So segmentation runs **once, offline, per location**, and its output is
not masks shipped to the browser — it is the **scene geometry**: per sound region an
azimuth, an elevation, an angular width and a distance.

At runtime, "analysis per viewport" is a frustum test against those angles. A few hundred
float operations at 20 Hz, no model, no WASM, no GPU. The client payload per location is a
~5 KB JSON file plus audio.

Because it is a build step, model choice optimises for quality, not speed. Twenty
inferences that take four minutes on a laptop are free — they happen once.

## Getting the whole sphere into a model

Segmentation models are trained on ordinary perspective photos. Equirectangular projection
stretches everything toward the poles — a patch of sky at the top of the frame becomes a
2048-pixel smear — and the masks smear with it.

`segment_panorama.py --mode multiview` (default) reprojects into ~20 overlapping 90–95°
perspective views (6 azimuths × 3 pitches, plus zenith and nadir), segments each, and
accumulates class probabilities back onto a sphere grid before taking the argmax. **The
overlap is what removes the seams.**

| Mode | Inferences | Trade-off |
| --- | --- | --- |
| `multiview` | ~20 | Best masks. Default. |
| `cube` | 6 | 3× faster, visible seams at face edges. |
| `erp` | 1 | Fastest, poles are mush. Fine if only the horizon band matters. |

## Partial panoramas

Most of our material is **not** a full 360. An iPhone sweep covers maybe 150°; a normal
photo covers 50–70°.

`embed_partial()` lifts the image onto a full equirectangular canvas, centred, carrying a
**validity mask** as a fourth channel. Everything downstream then works unchanged, and
regions can only form where there is actually image. Views that fall outside the covered
arc are skipped entirely rather than segmented as black.

The canvas is capped at 4096 px wide; a 153° sweep at 4800 px would otherwise need an
11294 × 5647 canvas, which is 191 MB of uint8 for no gain.

## Memory: the trap that reboots the machine

The model emits **150 channels** at roughly 128×128. If you upsample those to the input
resolution and collapse them to your own classes afterwards, the intermediate tensor is
`150 × H × W` floats. For a 6144×3072 equirect that is **11 GB**, and you pay it three
times — once for `interpolate`, once for `softmax`, once for the copy to host. ~34 GB.

On Apple unified memory that does not raise an OOM exception. The kernel swaps until the
watchdog reboots the machine. This happened once during development; it is the reason the
order below is not a style preference.

**Always softmax and collapse at the model's own resolution** (150 × 128 × 128 is 10 MB),
**then interpolate only the ~10 channels you keep.**

Two further guards are in the script:

- `--analysis-max-side` (default 1024) caps the resolution inference and accumulation run
  at. The model sees 512×512 regardless and the sphere grid is 512×256, so anything above
  this is memory spent for no accuracy — and the cost is quadratic. Views and their
  direction grids are downsampled together so nothing desyncs.
- A preflight line prints the actual cost before allocating: expect
  `~21 MB per view, 5 MB accumulator`. If it ever prints gigabytes, stop.

## Label set

ADE20K-150 covers most of an alpine scene. Verified indices from
`CSAILVision/sceneparsing/objectInfo150.csv`:

| Our sound class | ADE20K indices |
| --- | --- |
| sky | 3 |
| forest | 5 tree, 18 plant, 73 palm |
| pasture | 10 grass, 30 field, 67 flower |
| rock | 17 mountain, 35 rock, 69 hill |
| scree | 14 earth, 47 sand, 92 dirt track, 95 land |
| water | 22 water, 27 sea, 61 river, 129 lake |
| waterfall | 114 |
| trail | 53 path |
| built | 26 house, 33 fence, 49 skyscraper, 62 bridge |
| animal | 13 person, 127 animal |

ADE20K has **no** snow, glacier, scree-as-such, cattle or cable car. `--open-vocab` adds
those with CLIPSeg text prompts, summed onto the same accumulator as an ensemble. SAM 3
concept prompts would give better masks for the same job if there is time to wire it up.

Models: SegFormer-B4 ADE for speed, Mask2Former-Swin-L ADE for quality (~4× slower).

## Regions from the label map

Connected components per class on a 64×32 sphere grid, then per blob:

- **Solid angle** — sum of `cos(lat)·dlat·dlon` over its cells. Without the cosine
  weighting, the poles — a handful of real steradians — dominate every average.
- **Centroid direction** — averaged as a **vector**, not a scalar mean. A blob straddling
  the seam otherwise averages to the opposite side of the sphere.
- **Seam wrapping** — `ndimage.label` does not know the panorama wraps, so a waterfall on
  the seam comes back as **two half-loud sources on opposite sides**. We union them with
  a small union-find over the first and last columns.
- **Angular width** — radius of the equivalent spherical cap, `Ω = 2π(1 − cos r)`.
- **Distance** — ground-plane projection from 1.6 m eye height for anything below the
  horizon, taken from the region's **far edge** (85th percentile of elevation), not its
  centroid. The centroid of a big pasture sits at your feet; what you hear as "the pasture
  over there" is its far side. Above the horizon, a per-class default.

Blobs below a per-class solid-angle threshold are dropped — a 30-pixel scrap of water is
not a sound source.

Both the seam bug and the centroid-distance bug were caught by running the region
extractor against a synthetic label map before trusting it on a photo. Worth keeping that
habit: sphere geometry fails quietly and sounds wrong rather than crashing.

## Outputs

```
data/scenes/<id>/scene.segmented.json   engine-ready geometry
data/scenes/<id>/labels.png             the sphere label map
data/scenes/<id>/overlay.png            the photo with segmentation painted over it
```

`overlay.png` is the pitch asset: one image proving the sound placement is derived from
the picture rather than hand-placed. Put it on a slide even if the demo runs on
hand-tuned geometry.

## North

The script treats the image's **centre column as azimuth 0**. Set `north_offset_deg` in
`scene.json` to correct to true compass north. If the heading was not recorded at capture
it is unrecoverable — slate it on the recording next time.

## Honest advice for a 24-hour build

Hand-author the first location. Reading eight angles off a panorama takes ten minutes and
cannot fail at 4 a.m. Run `segment_panorama.py` in parallel on one image purely to produce
`overlay.png`, so the pipeline is *shown* while the demo the jury hears runs on numbers a
human typed. Ship the reliable thing; show the impressive thing.
