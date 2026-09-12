#!/usr/bin/env python3
"""
survey_recordings.py — work out what each take actually contains, before using any of it.

Two dozen files called R07_0021.WAV tell you nothing. This measures each one and says
what it probably is, so you can label the good ones and discard the unusable ones without
auditioning everything at 3 a.m.

    python survey_recordings.py                    # just look
    python survey_recordings.py --write-labels     # save the guesses for prepare_audio.py

What it measures, and why each one matters:

  <200 Hz share   Wind hitting the capsule is almost all low frequency. A take with
                  two-thirds of its energy under 200 Hz and a centroid near 400 Hz is
                  MIC RUMBLE, not the sound of wind in a landscape, and no amount of
                  filtering rescues it — high-passing just leaves you with quiet rumble.
  centroid        Where the energy sits. Running water is bright (>2 kHz); wind is dark.
  rolloff 85%     Confirms brightness independently of a few loud low-frequency frames.
  onsets/s        Sharp level jumps. Footsteps, handling noise, knocks.
  dB after HPF    How much survives a 100 Hz high-pass. A small drop means the content is
                  real; a large drop means the take WAS the rumble.

The guess is indicative, not authoritative — it is checked against the takes you already
labelled by hand, and it gets those right, but trust your ears over this table.

Needs: ffmpeg on PATH, numpy.
"""

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_IN = (REPO / "resources" / "recordings"
              if (REPO / "resources" / "recordings").is_dir()
              else REPO / "resources" / "sounds")
SR = 22050
AUDIO_EXT = (".wav", ".aiff", ".aif", ".flac", ".m4a", ".mp3", ".ogg")


def decode(path, hpf=None):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    if hpf:
        cmd += ["-af", f"highpass=f={hpf}:poles=2"]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True,
                         stdin=subprocess.DEVNULL, timeout=600).stdout
    return np.frombuffer(raw, dtype=np.float32)


def features(x):
    N, H = 1024, 512
    n = (len(x) - N) // H
    if n < 4:
        return None
    w = np.hanning(N).astype(np.float32)
    fr = np.lib.stride_tricks.as_strided(x, (n, N), (x.strides[0] * H, x.strides[0])) * w
    mag = np.abs(np.fft.rfft(fr, axis=1)) + 1e-10
    f = np.fft.rfftfreq(N, 1 / SR)

    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    cent = (mag * f).sum(1) / mag.sum(1)
    cum = np.cumsum(mag, 1) / mag.sum(1, keepdims=True)
    roll = f[np.argmax(cum >= 0.85, axis=1)]
    band = lambda lo, hi: mag[:, (f >= lo) & (f < hi)].sum(1) / mag.sum(1)
    d = np.diff(db, prepend=db[0])

    return dict(dur=len(x) / SR, db=float(db.mean()), std=float(db.std()),
                cent=float(np.median(cent)), roll=float(np.median(roll)),
                lo=float(band(0, 200).mean()), hi=float(band(2000, 8000).mean()),
                ons=float((d > 6).sum()) / (len(x) / SR),
                peak=float(np.abs(x).max()))


# guess -> (class for prepare_audio, usable?)
VERDICTS = {
    "running water":      ("water", True),
    "water, distant":     ("waterfall", True),
    "footsteps/handling": ("footsteps", True),
    "wind on the mic":    ("wind", False),
    "low ambience":       ("wind", True),
    "mixed ambience":     (None, True),
    "near silence":       (None, False),
}


def guess(r, h):
    """
    Order matters. Testing transients first makes running water read as footsteps —
    water has plenty of level variation. Spectrum first, then transients.
    """
    if h["cent"] > 1800 and h["roll"] > 3000:
        return "running water"
    if h["cent"] > 1100 and h["roll"] > 1800 and h["std"] < 8:
        return "water, distant"
    if r["lo"] > 0.55 and r["cent"] < 700:
        return "wind on the mic"
    if h["ons"] > 3 and h["std"] > 7:
        return "footsteps/handling"
    if r["db"] < -55:
        return "near silence"
    if h["cent"] < 900:
        return "low ambience"
    return "mixed ambience"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--write-labels", action="store_true",
                    help="write labels.json next to the recordings, which "
                         "prepare_audio.py reads instead of guessing from filenames")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"no recordings in {src}")

    print(f"{len(files)} recording(s) in {src}\n")
    print(f"  {'file':<28}{'dur':>6}{'dB':>7}{'<200Hz':>8}{'cent':>7}"
          f"{'| HPF100:':>10}{'dB':>7}{'cent':>7}{'roll':>7}{'ons/s':>7}   reading")

    rows, labels = [], {}
    for path in files:
        try:
            r = features(decode(path))
            h = features(decode(path, hpf=100))
        except Exception as e:
            print(f"  {path.name:<28} FAILED — {e}")
            continue
        if r is None or h is None:
            print(f"  {path.name:<28} too short to analyse")
            continue

        g = guess(r, h)
        cls, usable = VERDICTS[g]
        flag = "" if usable else "  <-- unusable"
        if r["peak"] > 0.99:
            flag += "  CLIPPED"
        print(f"  {path.name:<28}{r['dur']:>5.0f}s{r['db']:>7.1f}{r['lo']*100:>7.0f}%"
              f"{r['cent']:>7.0f}{'':>10}{h['db']:>7.1f}{h['cent']:>7.0f}{h['roll']:>7.0f}"
              f"{h['ons']:>7.1f}   {g}{flag}")
        rows.append((path, r, h, g, cls, usable))
        # Record EVERY take, usable or not. labels.json is the inventory: prepare_audio
        # needs to know a file is rubbish in order to skip it, and a take missing from
        # the file is indistinguishable from one nobody has looked at.
        labels[path.name] = {"class": cls, "guess": g, "usable": usable, "auto": True}

    usable = [x for x in rows if x[5] and x[4]]
    print(f"\n{len(usable)} of {len(rows)} takes look usable.")

    by_class = {}
    for _, _, _, _, cls, ok in rows:
        if cls and ok:
            by_class[cls] = by_class.get(cls, 0) + 1
    if by_class:
        print("  " + ", ".join(f"{k} x{v}" for k, v in sorted(by_class.items())))

    unusable = [x[0].name for x in rows if not x[5]]
    if unusable:
        print(f"\nFlagged unusable ({len(unusable)}): " + ", ".join(unusable))
        print("  Mostly wind on the capsule. A high-pass does not save these — the rumble")
        print("  IS the recording. Re-record with a proper windshield if you need wind.")

    if args.write_labels:
        dest = src / "labels.json"
        existing = {}
        if dest.exists():
            existing = json.loads(dest.read_text())
            # Never overwrite a label a human set by hand.
            labels = {k: v for k, v in labels.items()
                      if k not in existing or existing[k].get("auto")}
            merged = {**existing, **labels}
        else:
            merged = labels
        dest.write_text(json.dumps(dict(sorted(merged.items())), indent=2))
        good = sum(1 for v in merged.values() if v.get("usable"))
        print(f"\nwrote {dest}  ({len(merged)} take(s), {good} usable)")
        print("Edit it by hand where the guess is wrong — set \"auto\": false and a re-run")
        print("will leave your version alone. Then:")
        print("  python src/python/audio_prep/prepare_audio.py --scene <scene>")
    else:
        print("\nRe-run with --write-labels to save these for prepare_audio.py.")


if __name__ == "__main__":
    main()
