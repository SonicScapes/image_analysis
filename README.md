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
  recordings/               our own handheld-recorder WAVs from the site
  sounds-freesound/         CC0 downloads that fill classes we did not record
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
data/audio/                 processed stems, shared by every scene (never per-scene)
data/scenes/<id>/           generated: panorama.jpg, scene.json, overlay*.png
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

# SURVEY — what is actually in each take? Measures, guesses, writes labels.json.
python src/python/audio_prep/survey_recordings.py --write-labels

# PREPARE — our recordings in, seamless loops + one layer each out.
# --labelled-only uses just the takes whose filename says what they are; without it
# every recording becomes a simultaneous layer.
python src/python/audio_prep/prepare_audio.py --scene hohe-tauern --labelled-only

# serve the repo ROOT (not the app folder) and open the app
npm run dev
# http://localhost:3000/src/js/app/index.html
```

`ingest_images.py` writes into `data/scenes/hohe-tauern/`, so **use the same `--scene`
name for both** or you get two half-built scenes. Order does not matter: each merges into
`scene.json` instead of replacing it.

**Audio is not stored per scene.** `prepare_audio.py` writes stems to one shared pool at
`data/audio/`, and `scene.json` points at them with a relative path
(`../../audio/water.mp3`). This is safe because the engine never modifies a file: every
gain, pan, filter and distance decision happens at runtime from the numbers in
`scene.json`, and every stem is normalised to the same level so those numbers mean the
same thing everywhere. A stem used by five scenes is one file, encoded once — a re-run
for a second scene reports `reused` instead of spending the time again (`--force`
re-encodes).

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
| `<image>_classes.txt` | class coverage as integer percentages, `other` last — named after the source image |
| `overlay.png` | the photo with the labels painted over it |
| `overlay_labeled.png` | the same, with class names, shares and distances written on the regions — the pitch image |
| `labels.png` | the raw sphere label map |

The viewer can show any of these in place of the photo: the **Photo / Overlay / Named /
Labels** buttons in the top-left swap the displayed image while keeping the pan, which is
how you check whether `waterfall` really landed on a waterfall.

### Moving between scenes

The viewer reads `data/scenes/index.json` — a browser cannot list a directory over HTTP,
so `scene_index.py` writes it and every tool that creates a scene refreshes it. Run it by
hand after editing scenes yourself:

```bash
python src/python/scene_index.py
```

Then `‹ ›` in the top-left, the dropdown, or the **arrow keys** move between scenes, and
**1–4** switch the displayed image. A scene with no audio yet opens anyway, marked
`no audio` — you can pan and inspect it long before its sound exists.

Scenes open at a **75° field of view**, not fitted to the window. Fitting a 360 to the
window shows 288° at once, which is a picture *of* a panorama rather than standing in
one — and it flattens the audio, because nothing is ever out of view. Scroll to zoom.

### Exploring without sound

The gate has a second button, **Explore without sound**. It skips audio entirely — no
AudioContext, no downloads, no graph — but geometry, visibility, the meters and the
compass all still run. Use it to check placement before any audio exists, or when a scene
has no layers yet.

### Knowing what you recorded

Two dozen files called `R07_0021.WAV` tell you nothing.
`survey_recordings.py` measures each take — spectral centroid, low-frequency share,
onset rate, and what survives a 100 Hz high-pass — and says what it probably is. It
writes `labels.json` beside the recordings, which `prepare_audio.py` reads in preference
to guessing from filenames.

The measurement that matters most: **a take with two thirds of its energy below 200 Hz
and a centroid near 400 Hz is wind hitting the capsule, not wind in a landscape.** A
high-pass does not rescue it — the rumble *is* the recording. Those are marked
`usable: false` and skipped.

Edit `labels.json` by hand wherever the guess is wrong and set `"auto": false`; a re-run
leaves your version alone.

Two takes of the same class are handled differently by kind. A **region** keeps only the
longest take — four footstep loops from one direction is mud, and 6 dB too loud. An
**event** collapses into one layer holding every take as a variant, which is what stops a
cowbell sounding like a sample.

### Wiring scenes to audio

The two halves of the pipeline run past each other until this step. Segmentation knows
**where** things are but writes placeholder filenames; `prepare_audio.py` makes **real
audio** but guesses geometry from a filename hint table. `build_scene_layers.py` joins
them:

```
geometry  <- scene.segmented.json   (measured from the photograph)
audio     <- data/audio/            (our recordings first, downloads second)
```

```bash
python src/python/audio_prep/build_scene_layers.py --dry-run   # look first
python src/python/audio_prep/build_scene_layers.py
```

It writes the layers into each `scene.json` with the azimuths, elevations, spreads and
distances segmentation actually measured, and reports every class that has geometry but
no sound. A stem can stand in for a neighbouring class where that is honest — a wind bed
is what an exposed rock face sounds like, our footsteps take is what `person` sounds like
from two metres — and the layer records what it borrowed.

Mark a layer `"locked": true` and it survives untouched; everything else is regenerated,
so re-running after a better segmentation pass or a new batch of downloads simply
improves the scene.

### Do we have a sound for everything we can see?

```bash
python src/python/audio_prep/check_coverage.py --queries
```

Compares each scene's `<image>_classes.txt` against the layers wired up in its `scene.json`
and prints the gap, biggest share of the view first, with a Freesound query for each.
Classes that are silent by design (`sky`, `trail`) are reported as such, not as holes.

Then fill the gaps, driven by what the scenes actually contain rather than a guess:

```bash
python src/python/audio_prep/check_coverage.py --json   # -> data/audio-needs.json
node src/js/tools/fetch-sounds.mjs                      # fetches exactly those classes
```

`check_coverage.py --json` walks every scene's class report and exports the missing
classes with their queries, so the class taxonomy stays in one place (Python) rather than
drifting between two. Downloads are **source material**, so they land in
`resources/sounds-freesound/` beside our own takes — not in `data/scenes/`, which holds
only what the pipeline generates — and they go through the same `prepare_audio.py`:

```bash
export FREESOUND_TOKEN=xxxxxxxx        # https://freesound.org/apiv2/apply/
node src/js/tools/fetch-sounds.mjs

python src/python/audio_prep/prepare_audio.py \
    --in resources/sounds-freesound --scene hohe-tauern
```

Its query table is aimed at what segmentation actually found in the Hohe Tauern —
rock, snow, scree, glacier — not at the forest-and-lake set we assumed before shooting.
Files are named after the class they fill (`rock-1.mp3`, `snow-1.mp3`), and
`prepare_audio.py` recognises those names, so the layers come out positioned and tagged
without further work. Run it twice — once per source folder — and the second run merges
into the first.

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

## If the page loads but nothing works

`serve` rewrites URLs by default: it strips `.html`, then strips `/index`, landing the
page on `/src/js/app` with **no trailing slash**. Every relative path then resolves one
directory too high — `./app.js` becomes `/src/js/app.js` — so the module 404s, no click
handler is ever attached, and the Listen button does nothing.

`serve.json` in the repo root turns that off (`"cleanUrls": false`) — `serve` reads it
automatically, and there is no CLI flag for it in this version (`--no-clean-urls` is
rejected and the server refuses to start). If you launched the server before that file
existed, **restart it**. Any plain static server works too:

```bash
python3 -m http.server 3000
```

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

Our own recordings: ours. `fetch-sounds.mjs` pulls **CC0 and Attribution** by default — both are fine commercially,
and it writes `CREDITS.md` with every file, author, licence and link, which is all
Attribution asks for. `--cc0-only` narrows it to sounds needing no credit at all, at the
cost of far fewer results. For production-grade filler, the Sonniss
#GameAudioGDC bundle is free, royalty-free and commercial-use with no attribution — but
its licence forbids using the sounds to train AI, which matters if we ever go generative.
