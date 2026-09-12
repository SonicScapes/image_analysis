# SoundScapes

Hear a place before you go. A web app that shows a spot in the Hohe Tauern and mixes its
soundscape live from field recordings, following **where you are looking**.

Built for the Music & AI Hackathon 2026, Challenge 1 — *Experience Sound & Nature*
(cta / Austria Tourism). Intended product shape: a white-label module licensed to tourism
boards per trail.

## Structure

```
docs/                       what we decided and why — read architecture.md first
resources/                  source material, not in git
  photos/                   iPhone photos and sweep panoramas
  panoramas360/             raw Insta360 .insp — dual fisheye, rejected on purpose
  panoramas360_static/      Insta360 Studio equirect exports — use these
  sounds/                   handheld recorder WAVs from the site
src/
  python/                   offline pipeline — runs once per location
    image_analysis/
      ingest_images.py      any photo -> one geometry model (projection + field of view)
      segment_panorama.py   optional: a model writes the scene geometry for you
    audio_prep/
      prepare_audio.py      recorder WAVs -> normalised, seamless web loops
    requirements.txt
  js/                       runtime — runs in the browser
    engine/
      soundscape-engine.js  the mixer. No dependencies, no analysis, no model
    app/
      index.html, app.js    the viewer: pan the image, hear the mix follow
    tools/
      fetch-sounds.mjs      fill gaps from Freesound (CC0) for sounds we didn't record
data/scenes/<id>/           generated: panorama.jpg, *.mp3, scene.json
```

Python and JavaScript never call each other. Python writes `scene.json`; JavaScript reads
it. That file is the entire contract, which is why the two halves can be worked on by two
people at once.

## Build a scene

All Python runs in this project's own conda environment. It exists because the
segmentation step needs torch and opencv, both of which require numpy >= 2 — so it cannot
share an environment with anything pinned to numpy 1.26.4.

```bash
# once
conda env create -f environment.yml
conda activate soundscapes
cd /Users/tom/Code/Hackathons/SoundScapes

# SURVEY — what did we actually shoot, and where? Converts nothing.
python src/python/image_analysis/ingest_images.py --in resources/photos --catalog

# INGEST — photos in, panorama.jpg + the panorama block of scene.json out
python src/python/image_analysis/ingest_images.py --scene hohe-tauern

# ...or one point of interest per photo, as poi-1, poi-2, ...
python src/python/image_analysis/ingest_images.py --in resources/photos \
    --pick IMG_8753.HEIC IMG_8754.HEIC IMG_8755.HEIC IMG_8756.HEIC \
    --each --scene poi

# PREPARE — recorder WAVs in, seamless loops + one layer each out
python src/python/audio_prep/prepare_audio.py --scene hohe-tauern

# serve the repo ROOT (not the app folder) and open the app
npm run dev
# http://localhost:3000/src/js/app/index.html
```

Both write into `data/scenes/hohe-tauern/`, so **use the same `--scene` name for both** or
you get two half-built scenes. Order does not matter: each merges into `scene.json`
instead of replacing it.

Re-running is safe. `prepare_audio.py` keeps any `az` / `el` / `distance` you have tuned
by ear and only refreshes the audio files, so the loop is: listen, edit `scene.json`,
reload the page. That tuning is the actual authoring work — about ten minutes per location.

### Letting a model write the geometry

In the `soundscapes` environment, everything is already installed. First run also
downloads model weights (~250 MB for SegFormer-B4, ~700 MB more with `--open-vocab`),
cached in `~/.cache/huggingface` afterwards. On Apple Silicon it uses the MPS backend
automatically.

It prints its memory cost before allocating anything — expect `~21 MB per view, 5 MB
accumulator`. If that line ever shows gigabytes, kill it and read the memory section of
`docs/segmentation.md`.

```bash
python src/python/image_analysis/segment_panorama.py \
    data/scenes/hohe-tauern/panorama.jpg \
    --out data/scenes/hohe-tauern \
    --scene-json data/scenes/hohe-tauern/scene.json --open-vocab
```

For several points of interest at once:

```bash
for d in data/scenes/poi-*; do
  python src/python/image_analysis/segment_panorama.py "$d/panorama.jpg" \
      --out "$d" --scene-json "$d/scene.json" --open-vocab
done
```

Each scene folder then gains three files beside `scene.json`:

| File | What it is |
| --- | --- |
| `scene.segmented.json` | engine-ready geometry, plus a `classes` block. Never overwrites `scene.json` |
| `scene_classes.txt` | class coverage as integer percentages, `other` last |
| `overlay.png` | the photo with the labels painted over it — the pitch image |
| `labels.png` | the raw sphere label map |

`docs/image_categories.md` lists every category, where the lists live, and how to add one.
See `docs/segmentation.md` for what runs offline and the sphere-geometry traps — and read
its last section before deciding to run this tonight.

### What EXIF gives us for free

`--catalog` surveys every photo and groups the geotagged ones into candidate points of
interest, writing `data/photo-catalog.csv` and an OpenStreetMap link per cluster. Run it
before deciding which spot to build.

Two fields matter more than the rest:

- **`GPSImgDirection`** — the compass bearing the camera was pointing, which is exactly
  what `north_offset_deg` means to the viewer. Ingest writes it automatically, so azimuths
  become real bearings rather than offsets from the image centre. On a swept panorama it
  is the bearing at capture, so treat it as ±15° and nudge if a known landmark sits
  off-centre.
- **`FocalLengthIn35mmFilm`** — exact field of view, and it is present on sweep panoramas
  too. For a sweep the lens's long-edge angle is the *vertical* extent and
  `hfov = vfov × aspect`, which is far better than guessing from the aspect ratio alone
  (it put one of our panoramas at 170° where the guess said 153°).

`GPSAltitude` comes along too, and is the input the mix wants for thinning the spectrum
as a trail climbs.

## A note on the 360 files

A raw `.insp` off the camera is **two fisheye circles side by side**, not an
equirectangular frame. It has the same 2:1 aspect ratio, so it looks right and is
completely wrong — fed to the pipeline as equirectangular it produces nonsense geometry.
`ingest_images.py` detects and rejects it.

Export from Insta360 Studio instead. Those files land in `resources/panoramas360_static/`,
carry GPano metadata, and are detected exactly rather than guessed.

## The one idea

Every image is a **window onto a sphere** with a known horizontal field of view. A normal
photo is a small window, an iPhone sweep is a wide one, a 360 is the whole sphere. Once
`ingest_images.py` has written `projection` and `hfov_deg`, the viewer converts a pan
offset into a bearing and the audio engine never needs to know which camera was used.

Every sound is a **direction, a width and a distance** on that same sphere. The engine
compares the two, forty times a second, and rebalances the mix.

No model runs at runtime. Nothing analyses pixels in the browser. The whole thing is a few
hundred float operations at 20 Hz, so it runs on a four-year-old phone and works offline
once loaded.

## Two behaviours that look like bugs and are not

**Nothing ever mutes.** Looking at the waterfall lifts it about 10 dB against a floor.
Turning away does not silence it, because in a real place you hear what is behind you.

**The mix lags slightly behind the drag.** 360 ms to rise, 1.8 s to fall. Instant response
sounds like a switch; this sounds like a place.

## Docs

| File | What's in it |
| --- | --- |
| `docs/architecture.md` | The system, the decisions, and what we deliberately did not build |
| `docs/challenge-fit.md` | How this answers cta's three challenge questions — read before the pitch |
| `docs/segmentation.md` | Image analysis: what runs offline, which classes, the sphere-geometry traps |
| `docs/image_categories.md` | Every recognisable category, where the lists live, how to add one |
| `docs/demo-script.md` | The two-minute jury run, beat by beat |

## Licensing

Our own recordings: ours. `fetch-sounds.mjs` pulls **CC0 only** by default and writes
`CREDITS.md` for whatever it took. For production-grade filler, the Sonniss
#GameAudioGDC bundle is free, royalty-free and commercial-use with no attribution — but
its licence forbids using the sounds to train AI, which matters if we ever go generative.
