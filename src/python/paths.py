"""
paths.py — the one place that knows where the generated data lives.

The viewer owns what comes out of this one: this folder holds the sources (resources/)
and the code that turns them into scenes, 360viewer_app holds the scenes themselves. So
everything the pipeline writes lands under DATA_ROOT, which sits next door.

The default is relative — image_analysis and 360viewer_app are siblings under one
project root, so a clone works wherever it is put, with nothing to edit:

    <anywhere>/image_analysis                     this folder
    <anywhere>/360viewer_app/data                 DATA_ROOT

Set SOUNDSCAPES_DATA to override it — a second export, a scratch run, a checkout laid
out differently. Every script takes an explicit --out/--scenes/--audio too; this is only
what they fall back to.

(Before the two were folded into one project root, they lived as separate sibling
repos named SoundScapes and SonicScapes, and this default used to say
`REPO.parent / "SonicScapes" / "360viewer_app" / "data"`. After the move that doubles
the "SonicScapes" segment and points nowhere — fixed here to just go one level up.)
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get(
    "SOUNDSCAPES_DATA",
    REPO.parent / "360viewer_app" / "data",
))

SCENES_DIR = DATA_ROOT / "scenes"
AUDIO_DIR = DATA_ROOT / "audio"
