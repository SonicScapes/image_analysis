"""
paths.py — the one place that knows where the generated data lives.

The viewer is its own repo now and owns what comes out of this one: this repo holds the
sources (resources/) and the code that turns them into scenes, the viewer repo holds the
scenes themselves. So everything the pipeline writes lands under DATA_ROOT, which sits
next door.

The default is relative — the two repos are siblings, so a clone of the pair works
wherever it is put, with nothing to edit:

    <anywhere>/SoundScapes                        this repo
    <anywhere>/SonicScapes/360viewer_app/data     DATA_ROOT

Set SOUNDSCAPES_DATA to override it — a second export, a scratch run, a checkout laid
out differently. Every script takes an explicit --out/--scenes/--audio too; this is only
what they fall back to.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get(
    "SOUNDSCAPES_DATA",
    REPO.parent / "SonicScapes" / "360viewer_app" / "data",
))

SCENES_DIR = DATA_ROOT / "scenes"
AUDIO_DIR = DATA_ROOT / "audio"
