# Image categories

The list of things we can recognise in a photo, where each list lives, and what to touch
when you add one.

Everything is in **`src/python/image_analysis/segment_panorama.py`**, in four tables near
the top of the file. They are separate on purpose: what we can *see* and what we can
*hear* are different questions, and one class can answer the first without answering the
second.

| Table | Answers | Used by |
| --- | --- | --- |
| `ALPINE_CLASSES` | which ADE20K classes collapse into each of ours | segmentation |
| `OPEN_VOCAB_PROMPTS` | classes ADE20K has no label for, via CLIPSeg text prompts | `--open-vocab` |
| `SOUND_SPEC` | whether a class makes a sound, and how it behaves | scene geometry |
| `QUERIES` | what to search Freesound for, if we didn't record it | `fetch-sounds.mjs` |
| `PALETTE` | the colour it gets in `overlay.png` / `labels.png` | the pitch image |

## The categories

### From ADE20K (always available)

ADE20K's 150-class set is a general scene-parsing vocabulary, so several of its labels
collapse into one thing we care about. Indices are verified against
`CSAILVision/sceneparsing/objectInfo150.csv` and are 1-based, as that file is.

| Our class | ADE20K labels folded in | Sounds? |
| --- | --- | --- |
| `sky` | 3 sky | no — the wind bed covers the sky |
| `forest` | 5 tree, 18 plant, 73 palm | yes, region |
| `pasture` | 10 grass, 30 field, 67 flower | yes, region |
| `rock` | 17 mountain, 35 rock, 69 hill | yes, region |
| `scree` | 14 earth, 47 sand, 92 dirt track, 95 land | yes, region |
| `water` | 22 water, 27 sea, 61 river, 129 lake | yes, region |
| `waterfall` | 114 waterfall | yes, region — loudest thing in the set |
| `trail` | 53 path | no — informational only |
| `built` | 26 house, 33 fence, 49 skyscraper, 62 bridge | yes, region |
| `animal` | 127 animal | yes, event — wildlife |
| `person` | 13 person | yes, event — tagged `human` |

### From CLIPSeg text prompts (`--open-vocab`)

ADE20K has no label for any of these, and an alpine scene is full of them. Each is a
sentence handed to CLIPSeg, whose output is summed onto the same accumulator as an
ensemble with the ADE pass.

| Our class | Prompt | Sounds? |
| --- | --- | --- |
| `snow` | "a snowfield on a mountain" | yes, region |
| `glacier` | "a glacier of blue ice" | yes, region |
| `scree` | "a slope of loose grey scree and broken rock" | yes, region — also ADE-backed |
| `cattle` | "cows grazing on an alpine pasture" | yes, event |
| `cablecar` | "a cable car line or ski lift pylon" | yes, event, tagged human |
| `person` | "people hiking, walkers with backpacks" | yes, event — also ADE-backed |

### `other`

Not a category — a residual, and a deliberately meaningful one.

The ADE model spreads probability across all 150 of its classes. We map about ten of
them. Whatever mass lands on the other 140 is reported as `other`: *the model saw
something and we have no category for it*. No threshold, no tuning.

So `other` is a diagnostic. A few percent is normal. **Thirty percent means the label set
is missing something that is actually in front of the camera** — open `overlay.png`, find
the grey, and add a class.

### Why `person` is not `animal`

ADE20K's label 13 is `person`, and it was originally folded into our `animal` class
because both are "a moving thing that makes a noise". That was wrong twice over: a hiker
would trigger a bird call, and the one class the human-pressure control exists to notice
could not be seen at all.

`person` is now its own class, an event tagged `human`, and it does something the others
do not: **a detected person raises the scene's starting pressure rather than being hidden
by it.** `segment_panorama.py` writes `scene.pressure` from what is visibly human —

```
pressure = 2 * sqrt(person% + 0.8 x cablecar% + 0.5 x built%)      capped at 10
```

— and the engine and viewer start there instead of at pristine. The square root is
deliberate: one hiker in frame changes how a place feels far more than their 1 % of the
pixels suggests. If there are people in the picture, the place already has people in it,
and pretending otherwise is the kind of flattery this project is supposed to avoid.

## `scene_classes.txt`

Written next to `scene.json` on every segmentation run, and mirrored into
`scene.segmented.json` under `classes` so code needn't parse the text.

```
# class coverage of panorama.jpg (multiview pass)
# solid-angle weighted over the photographed area, so these are shares of what
# the camera saw — not pixel counts, and not shares of the whole sphere.
# 'other' is probability the model gave to classes we do not map.

pasture 42%
rock    31%
sky     18%
scree   3%
other   6%
```

Three things about those numbers:

- **Solid angle, not pixels.** An equirectangular frame gives the zenith as many pixels
  as the horizon, so a pixel count reports every scene as mostly sky. `cos(latitude)`
  weighting fixes it, and it is the same weighting the sound regions use.
- **Of the photographed area only.** For a 150° sweep the missing 210° is not `other`, it
  is simply not in the picture.
- **They sum to 100.** Integer percentages by largest remainder; classes under 1 % are
  folded into `other` rather than printed as a `0 %` row.

## Adding a category

Say you want `bog` — the Hohe Tauern has plenty, and it sounds distinctive.

1. **See it.** Either add ADE indices to `ALPINE_CLASSES["bog"]`, or add a prompt to
   `OPEN_VOCAB_PROMPTS["bog"]` if ADE has nothing close. Prompts want a descriptive
   sentence, not a noun: *"boggy ground with dark standing water and tussock grass"*.
2. **Hear it.** Add a `SOUND_SPEC["bog"]` row: `(type, gain_db, focus_db, far_m,
   min_solid_angle_sr)`. `type` is `"region"`, `"event"`, or `None` for a class that is
   recognised but silent. `min_solid_angle_sr` is the size below which a blob is ignored —
   a 30-pixel scrap of bog is not a sound source.
3. **Source it.** Add `QUERIES["bog"]` so `fetch-sounds.mjs` can fill the gap if we never
   recorded one.
4. **Colour it.** Add `PALETTE["bog"]` so it is distinguishable in `overlay.png`. Pick
   something no neighbouring class already uses, or the pitch image gets harder to read.

Steps 1 and 4 are the minimum to make it appear in `scene_classes.txt`. Without step 2 it
is seen and reported but never sounds.

## The audio side has its own tags

Separate from these categories, layers carry `tags` that drive the mixer's macro
controls. They are assigned in `prepare_audio.py`'s `HINTS` table, keyed off the recording
filename, not off image segmentation:

| Tag | Effect |
| --- | --- |
| `wind` | level follows the scenicness axis |
| `wildlife` | event rate drops up to 85 % as human pressure rises |
| `human` | held 40 dB down until human pressure rises |

A class and a tag are different things: `cablecar` is something we can *see*, while
`human` is something a *sound* is. The lift pylon in the image and the lift hum in the
recording are related but not the same object, and the code keeps them apart.
