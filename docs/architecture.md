# Architecture

v0.2 — 12 Sep 2026. Supersedes the pre-hackathon draft where they disagree.

## What the product is

A web module that shows a spot on a trail and mixes its soundscape live from field
recordings, following where the visitor looks. Sold **white-label to tourism boards, per
trail** — they already own the imagery and the GPX; we add the layer nobody else sells,
which is what the place sounds like to stand in.

That business shape decides two technical things:

- **The imagery and tour data come from the client**, so we never need Komoot or Google
  data rights. Our pipeline has to swallow whatever they hand us — a phone photo, a sweep
  panorama, a 360 — which is why `ingest_images.py` normalises all three.
- **Content is auditioned and signed off** before it ships. Nothing a visitor can hear may
  be generated on the fly, because a client cannot approve what does not exist yet.

## The load-bearing decision: nothing generative at runtime

No model runs in the request path. Not for audio, not for image analysis.

| Constraint | Reality |
| --- | --- |
| Latency | Text-to-audio returns a 30 s clip in 3–20 s. The viewport must respond inside ~200 ms. |
| Cost | Per-second generation pricing scales with engagement — the wrong shape for a free tourism funnel. |
| Consistency | Two generations from one prompt differ in level and tone. A place must sound like itself every visit. |
| Review | The client signs off on content. It has to exist to be approved. |
| Reception | Trails have no signal. The scene must run with the network off. |

So all audio is recorded or licensed, then normalised and human-checked at build time as
loopable stems. The runtime is a deterministic Web Audio graph — gains, filters, HRTF
panners, one reverb — costing a few hundred float operations at 20 Hz.

Generativity lives in two other places, where it is free:

- **Authoring**: models write the scene *geometry* offline (`segment_panorama.py`) and can
  write filler stems we failed to record.
- **Arrangement**: a stochastic scheduler decides which one-shot fires when and from
  where, so no two visits play the same sequence without a model in the loop.

## Two geometries, one contract

Everything rests on describing image and sound in the same coordinate system.

**Images.** Every photo is a window onto a sphere with a known horizontal field of view.
`ingest_images.py` writes `projection` (`flat` / `cylindrical` / `equirect`) and
`hfov_deg`, detected from GPano XMP where present, else EXIF focal length, else aspect
ratio — and it says which, so a guess is visible rather than silent. A normal photo is a
small window you can pan a little inside; a 360 is the whole sphere. Same code path.

**Sounds.** Every layer is a direction (`az`, `el`), an angular width (`spread`) and a
distance in metres. Four kinds:

| Layer | Material | Spatialisation | Driven by |
| --- | --- | --- | --- |
| `bed` | The site recording, looped | Non-directional, always on | Altitude and exposure |
| `region` | Per class: waterfall, forest, scree, pasture | Wide, weighted toward its centroid | Visible fraction of that region |
| `event` | One-shots: bird, cowbell, rockfall, footstep | HRTF point source, distance-attenuated | Stochastic scheduler × visibility |
| `score` | Tonal pad, optional | Centred, non-diegetic | Scenicness; ducked under events |

`data/scenes/<id>/scene.json` holds both geometries and is the only contract between the
Python pipeline and the JavaScript runtime.

## Runtime, per frame

1. **Visibility** — elliptical frustum test per layer against its angular disc, `0…1`.
2. **Gain** — `base + distanceDb − focus·(1 − visibility)`, floored at −15 dB.
3. **Smoothing** — 360 ms to rise, 1.8 s to fall.
4. **Distance** — `−20·log10(d/20)` *and* a low-pass, so far things are quiet **and** dull.
5. **Panning** — HRTF panner per source; the listener rotates, sources stay in world space.
6. **Events** — rate scales with visibility, ±6 % pitch variation per trigger.
7. **Macros** — scenicness and pressure set spectral tilt, reverb, wind, event density.

### Two deliberate behaviours

**Nothing ever mutes.** Looking at a source lifts it 6–10 dB against a floor. A hard gate
on what leaves the viewport reads as a broken mute button, not as immersion — you hear what
is behind you in real life.

**Interpolate parameters, cross-fade material.** Moving between locations morphs the
control vector continuously and cross-fades stems with equal loudness. Never morph audio
content itself; the result sounds like neither place and no tuning fixes it.

## The two expressive axes

From **ISO 12913-3**'s circumplex of soundscape perception, so the scale is defensible
rather than invented:

- **Scenicness** (pleasant ↔ annoying) — spectral tilt, reverb size, wind level, score.
  Its low end is **austere, exposed, vast**, never ugly. A tourism client will not license
  a page that makes their glacier sound bad.
- **Human pressure** (0–10) — raises layers tagged `human` from 40 dB down to their
  authored level, drops `wildlife` event rates by up to 85 %, and trims reverb because a
  busy place sounds smaller. This is what answers the brief's sustainability question; see
  `challenge-fit.md`.

## Stack

| Concern | Choice | Why |
| --- | --- | --- |
| Viewer | 2D CSS transform over the image | A window onto a sphere needs no WebGL. Nothing to fail on stage. |
| Audio | Web Audio API, hand-built graph | HRTF panners and filters are built in. No library. |
| Delivery | MP3 stems, fully preloaded | ~3–5 MB per location, then no network. |
| Overview map (later) | MapLibre + Austrian open basemap/DTM | Free, Austria-correct, no metering. Google 3D Tiles restricts caching and derivation. |
| Distances (later) | DTM raycast from the known camera position | Real metres beat a monocular depth guess everywhere the terrain model covers. |

## What we deliberately did not build

- **Runtime segmentation.** A static image segmented per viewport recomputes an unchanging
  answer at 30 fps. See `segmentation.md`.
- **Full 360 video.** 20–35 Mbps, hostile to mobile data and battery. Stills plus short
  looping insets deliver most of the life.
- **Ambisonic beds.** The recorders are stereo. First-order ambisonics with rotation is the
  upgrade once we record with an ambisonic mic; the engine has a slot for it.
- **A CMS.** Scene authoring is a JSON file and ten minutes of listening. Tooling can wait
  until a client asks for self-service.

## Known gaps

- **Compass heading.** Azimuths are relative to the image centre unless `heading_deg` came
  from metadata. `north_offset_deg` corrects it, but the true bearing is unrecoverable if
  it was not noted at capture — slate it on the recording next time.
- **Loudness normalisation is RMS-based**, not true LUFS. Fine for consistency between our
  own stems; revisit before mixing in third-party material.
- **Scenicness is not yet predicted** from the image — it is a hand-set number. The CLIP +
  Scenic-Or-Not regressor is the real version.
- **Distances above the horizon are class defaults.** The DTM raycast replaces them.
