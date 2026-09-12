#!/usr/bin/env python3
"""
ingest_images.py — normalise any photo into the one geometry model the app understands.

We shoot three different things and they must all end up describable the same way:

  regular iPhone photo   ~68° horizontal, flat perspective   ->  projection "flat"
  iPhone sweep panorama  ~120-220° horizontal, cylindrical   ->  projection "cylindrical"
  Insta360 .insp         360° x 180°, equirectangular        ->  projection "equirect"

The unifying idea: every image is a WINDOW ONTO A SPHERE with a known horizontal field
of view. Once we know `projection` and `hfov_deg`, the viewer can convert a pan offset
into a bearing, and the audio engine needs no knowledge of which camera took the shot.
A regular photo simply has a small window you can pan a little way inside; a 360 has a
window you can pan all the way around. Same code path, one number different.

    python ingest_images.py --catalog                      # survey everything first
    python ingest_images.py --pick IMG_8800.HEIC --scene poi-2

Detection order, most reliable first:
  1. GPano XMP metadata (written by 360 cameras and by some panorama apps) — exact geometry.
  2. .insp / .insv extension, or a 2:1 aspect ratio at 360-camera resolution — equirectangular.
  3. EXIF FocalLengthIn35mmFilm — exact for a normal photo AND for a sweep panorama, where
     the lens's long-edge angle is the VERTICAL extent and hfov = vfov x aspect ratio.
  4. Aspect ratio heuristics — last resort, and it says so in the output.

EXIF also gives us, for free:
  - GPSImgDirection — the compass bearing the camera was pointing, which is precisely
    what north_offset_deg means. Azimuths become real bearings instead of being
    relative to the image centre.
  - GPSLatitude/Longitude/Altitude — position and metres above sea level, so shots can
    be grouped into points of interest (--catalog) and the altitude can drive the mix.

Needs: pillow, pillow-heif (for HEIC). numpy only for the fisheye check.
"""

import argparse, json, math, re, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

REPO = Path(__file__).resolve().parents[3]
# Prefer the stitched 360 exports; raw .insp files are dual fisheye and get rejected.
DEFAULT_IN = (REPO / "resources" / "panoramas360_static"
              if (REPO / "resources" / "panoramas360_static").is_dir()
              else REPO / "resources" / "photos")
DEFAULT_OUT = REPO / "scenes"

# Web delivery caps. A 7000 px JPEG is 5 MB and nobody can see the difference while
# panning; these keep a scene under ~2 MB of imagery.
MAX_WIDTH = {"equirect": 6144, "cylindrical": 4800, "flat": 2400}
JPEG_QUALITY = 84
SENSOR_MM = 36.0  # 35 mm equivalent frame width


def _refresh_index(scenes_dir):
    """The viewer cannot list a directory over HTTP, so keep an index file current."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scene_index import write_index
        dest, entries = write_index(scenes_dir)
        print(f"index: {len(entries)} scene(s) -> {rel(dest)}")
    except Exception as e:
        print(f"note: could not write the scene index ({e})")


def slug(name):
    """Folder-and-URL-safe version of a filename stem. Scene ids end up in URLs."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return out or "scene"


def category_dir(projection):
    """Which subfolder a scene of this projection lives in, under the scenes root.

    A 360 (equirect, or a cylindrical sweep — both something you turn all the way around
    inside) is a different kind of asset from a flat photo, and keeping them apart makes
    the scenes root browsable by eye instead of one long alphabetical list of IMG_ and
    poi- names.
    """
    return "360pano" if projection in ("equirect", "cylindrical") else "photos"


def find_scene_dir(root, scene_id):
    """An existing scene, wherever its category subfolder put it — or, if there is no
    such scene yet, None (the caller decides where a new one belongs)."""
    hits = [p for p in root.glob(f"*/{scene_id}")
            if p.is_dir() and p.parent.name != "audio"]
    return hits[0] if hits else None


def rel(p):
    """Path relative to the repo when it is inside it, absolute otherwise."""
    try:
        return Path(p).relative_to(REPO)
    except ValueError:
        return Path(p)


# ---------------------------------------------------------------- metadata
def read_gpano(path):
    """
    Pull GPano XMP fields out of the raw bytes. Works for JPEG and for .insp (which is a
    JPEG wearing a different extension). GPano gives exact partial-panorama geometry, so
    when it is present we don't guess anything.
    """
    try:
        # Bounded read. read_bytes()[:N] pulls the entire file first, which on a 35 MB
        # panorama — or anything living on a synced/remote volume — is the slow part.
        with open(path, "rb") as fh:
            blob = fh.read(2_000_000)
    except OSError:
        return {}
    text = blob.decode("latin-1", errors="ignore")
    if "GPano" not in text:
        return {}
    keys = ["ProjectionType", "FullPanoWidthPixels", "FullPanoHeightPixels",
            "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels",
            "CroppedAreaLeftPixels", "CroppedAreaTopPixels", "PoseHeadingDegrees"]
    out = {}
    for k in keys:
        m = re.search(rf'GPano:{k}\s*=\s*"([^"]+)"', text) or \
            re.search(rf"<GPano:{k}>([^<]+)</GPano:{k}>", text)
        if m:
            v = m.group(1).strip()
            try:
                out[k] = float(v) if k != "ProjectionType" else v
            except ValueError:
                out[k] = v
    return out


def read_exif(img):
    """
    Note: FocalLengthIn35mmFilm lives in the Exif sub-IFD (0x8769), not the top-level
    one, so getexif() alone never finds it. Merge both.
    """
    out = {}
    try:
        raw = img.getexif()
    except Exception:
        return out
    merged = dict(raw)
    try:
        merged.update(dict(raw.get_ifd(0x8769)))
    except Exception:
        pass
    names = {v: k for k, v in ExifTags.TAGS.items()}
    for key in ("FocalLengthIn35mmFilm", "FocalLength", "Make", "Model",
                "DateTimeOriginal", "LensModel"):
        tag = names.get(key)
        if tag and tag in merged:
            out[key] = merged[tag]

    # GPS sits in its own IFD (0x8825). iPhones fill it generously.
    try:
        gps = raw.get_ifd(0x8825)
    except Exception:
        gps = {}
    if gps:
        g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
        lat = _dms(g.get("GPSLatitude"), g.get("GPSLatitudeRef"), "S")
        lon = _dms(g.get("GPSLongitude"), g.get("GPSLongitudeRef"), "W")
        if lat is not None and lon is not None:
            out["lat"], out["lon"] = lat, lon
        if g.get("GPSAltitude") is not None:
            alt = float(g["GPSAltitude"])
            # GPSAltitudeRef 1 means below sea level.
            ref = g.get("GPSAltitudeRef")
            if isinstance(ref, bytes) and ref and ref[0] == 1:
                alt = -alt
            out["elev_m"] = alt
        # The prize: the compass bearing the camera was actually pointing. This is what
        # makes north_offset_deg recoverable instead of guessed.
        if g.get("GPSImgDirection") is not None:
            out["heading"] = float(g["GPSImgDirection"]) % 360
            out["heading_ref"] = g.get("GPSImgDirectionRef", "T")   # T = true, M = magnetic
        if g.get("GPSHPositioningError") is not None:
            out["gps_error_m"] = float(g["GPSHPositioningError"])
        if g.get("GPSDateStamp") and g.get("GPSTimeStamp"):
            h, m, sec = [float(x) for x in g["GPSTimeStamp"]]
            out["utc"] = f"{g['GPSDateStamp'].replace(':', '-')} {int(h):02d}:{int(m):02d}:{int(sec):02d}Z"
    return out


def _dms(value, ref, negative_ref):
    """EXIF degrees/minutes/seconds triple -> signed decimal degrees."""
    if not value:
        return None
    try:
        d, m, s = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    dec = d + m / 60 + s / 3600
    return -dec if ref == negative_ref else dec


def looks_like_dual_fisheye(img):
    """
    Some 360 cameras hand you two circular fisheye images side by side instead of a
    stitched equirectangular frame — same 2:1 aspect, completely different geometry, and
    feeding one to the pipeline as if it were equirectangular produces nonsense geometry.

    Circular images leave the four corners black. An equirectangular frame cannot: its top
    row is the zenith and its bottom row the nadir, both stretched all the way across, so
    the corners carry sky and ground. Corners alone are therefore a strong discriminator —
    we take the median so one bright lens flare can't outvote the rest.
    """
    a = np.asarray(img.convert("L").resize((256, 128)), dtype=np.float32)
    h, w = a.shape
    patch = 16
    corners = [a[:patch, :patch], a[:patch, w - patch:],
               a[h - patch:, :patch], a[h - patch:, w - patch:]]
    corner_med = float(np.median([c.mean() for c in corners]))

    # Second signal: in a side-by-side fisheye the seam between the two circles is dark
    # top and bottom. NOTE the x coordinate is the image midpoint (w // 2), not h // 2 —
    # getting that wrong samples bright sky inside the left circle and the check fails.
    mid = w // 2
    seam = [a[:patch, mid - patch // 2:mid + patch // 2],
            a[h - patch:, mid - patch // 2:mid + patch // 2]]
    seam_med = float(np.median([c.mean() for c in seam]))

    return corner_med < 30.0 and seam_med < 60.0


# ---------------------------------------------------------------- geometry
def classify(path, img):
    """
    Work out how this image maps onto the sphere.

    Returns a dict: projection, hfov_deg, vfov_deg, how (what told us), plus whatever
    EXIF had to say about position, altitude and compass heading.
    """
    w, h = img.size
    ratio = w / h
    gp = read_gpano(path)
    ex = read_exif(img)

    meta = {k: ex[k] for k in ("lat", "lon", "elev_m", "heading", "heading_ref",
                               "gps_error_m", "utc", "Model", "LensModel")
            if k in ex}

    def out(projection, hfov, vfov, how, heading=None):
        d = dict(meta)
        d.update(projection=projection, hfov_deg=hfov, vfov_deg=vfov, how=how)
        if heading is not None:
            d["heading"] = heading % 360
            d.setdefault("heading_ref", "T")
        return d

    # 1. GPano — exact geometry, written by 360 cameras and some panorama apps.
    if gp.get("ProjectionType", "").lower().startswith("equirect") or "FullPanoWidthPixels" in gp:
        full_w = gp.get("FullPanoWidthPixels", w)
        full_h = gp.get("FullPanoHeightPixels", h)
        crop_w = gp.get("CroppedAreaImageWidthPixels", w)
        crop_h = gp.get("CroppedAreaImageHeightPixels", h)
        hfov = 360.0 * crop_w / full_w
        vfov = 180.0 * crop_h / full_h
        proj = "equirect" if hfov > 350 else "cylindrical"
        return out(proj, hfov, vfov, "GPano metadata", gp.get("PoseHeadingDegrees"))

    # 2. A 360 camera file.
    if path.suffix.lower() in (".insp", ".insv") or (abs(ratio - 2.0) < 0.04 and w >= 4000):
        if looks_like_dual_fisheye(img):
            return out("dual_fisheye", 360.0, 180.0, "dual fisheye — needs stitching")
        return out("equirect", 360.0, 180.0,
                   ".insp/.insv file" if path.suffix.lower() in (".insp", ".insv")
                   else "2:1 aspect at 360-camera resolution")

    # 3. EXIF focal length — exact, and it is present on sweep panoramas too.
    f35 = ex.get("FocalLengthIn35mmFilm")
    if f35:
        f35 = float(f35)
        # The 35 mm equivalent describes the LONG edge of the frame.
        long_fov = math.degrees(2 * math.atan(SENSOR_MM / (2 * f35)))
        if ratio >= 2.2:
            # A sweep panorama is shot in portrait and swept sideways, so the lens's
            # long-edge angle becomes the VERTICAL extent and the horizontal extent is
            # however far the phone was turned. Apple crops a few percent for
            # stabilisation, hence the 0.96.
            vfov = long_fov * 0.96
            hfov = min(vfov * ratio, 360.0)
            how = f"EXIF {f35:.0f} mm equivalent, swept {hfov:.0f}deg"
            return out("cylindrical", hfov, vfov, how)
        hfov = long_fov if ratio >= 1 else long_fov * ratio
        return out("flat", hfov, hfov / ratio, f"EXIF {f35:.0f} mm equivalent")

    # 4. Aspect heuristics — last resort, and it says so.
    if ratio >= 2.2:
        hfov = min(360.0, 60.0 * ratio)
        return out("cylindrical", hfov, hfov / ratio,
                   f"guessed from {ratio:.2f}:1 aspect — check this")
    long_fov = 68.0
    hfov = long_fov if ratio >= 1 else long_fov * ratio
    return out("flat", hfov, hfov / ratio, "assumed 68deg phone lens — check this")


def haversine_m(a, b):
    """Metres between two (lat, lon) pairs. For clustering photos into PoIs."""
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(x)))


def cluster_by_location(views, radius_m=60.0):
    """Greedy grouping of geotagged shots into candidate points of interest."""
    clusters = []
    for v in views:
        if "lat" not in v:
            continue
        here = (v["lat"], v["lon"])
        for c in clusters:
            if haversine_m(here, c["centre"]) <= radius_m:
                c["views"].append(v)
                n = len(c["views"])
                c["centre"] = (sum(x["lat"] for x in c["views"]) / n,
                               sum(x["lon"] for x in c["views"]) / n)
                break
        else:
            clusters.append({"centre": here, "views": [v]})
    clusters.sort(key=lambda c: -len(c["views"]))
    return clusters


# ---------------------------------------------------------------- output
def convert(path, img, projection, out_dir, stem):
    w, h = img.size
    cap = MAX_WIDTH.get(projection, 4096)
    if w > cap:
        img = img.resize((cap, round(h * cap / w)), Image.LANCZOS)
    dest = out_dir / f"{stem}.jpg"
    img.convert("RGB").save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True,
                            progressive=True)
    return dest, img.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--scene", default="hohe-tauern")
    ap.add_argument("--pick", nargs="*", default=None,
                    help="only these filenames (default: every image in --in)")
    ap.add_argument("--name", default="panorama",
                    help="output stem; the scene's main view (default: panorama)")
    ap.add_argument("--heading", type=float, default=None,
                    help="override the compass bearing of the image CENTRE")
    ap.add_argument("--each", action="store_true",
                    help="treat every input image as its own point of interest and write "
                         "one scene folder per image, NAMED AFTER THE INPUT FILE "
                         "(IMG_1234.jpg -> scenes/360pano/IMG_1234/ or scenes/photos/IMG_1234/, "
                         "by its own projection)")
    ap.add_argument("--redo", action="store_true",
                    help="re-ingest images whose scene has already been segmented. "
                         "Without it they are skipped, since re-converting a panorama "
                         "that is already done is pure cost.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N images, in sorted filename order")
    ap.add_argument("--number", action="store_true",
                    help="with --each, name folders <scene>-1, <scene>-2 ... instead of "
                         "after the input file")
    ap.add_argument("--catalog", action="store_true",
                    help="survey every image (position, altitude, bearing, field of view), "
                         "group them into candidate points of interest, write a CSV, and "
                         "convert nothing")
    ap.add_argument("--cluster-radius", type=float, default=60.0,
                    help="metres within which shots count as the same place (%(default)s)")
    args = ap.parse_args()

    src = Path(args.src)

    exts = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".insp", ".insv", ".dng", ".tif")
    files = ([src / p for p in args.pick] if args.pick else
             sorted(p for p in src.iterdir() if p.suffix.lower() in exts))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no images found in {src}")

    # Skip anything already segmented. The scene id comes from the filename, so this
    # costs nothing — no need to open the image to find out we don't want it. The scene
    # could be filed under either category subfolder (its projection isn't known yet,
    # here — that needs the image open), so look for it under both.
    skipped_done = []
    if not args.redo:
        kept = []
        for f in files:
            scene_id = slug(f.stem) if args.each else args.scene
            existing = find_scene_dir(Path(args.out), scene_id)
            if existing and (existing / "scene.segmented.json").exists():
                skipped_done.append(scene_id)
            else:
                kept.append(f)
        files = kept
    if skipped_done:
        print(f"skipping {len(skipped_done)} already segmented: "
              + ", ".join(skipped_done[:4])
              + (f" … (+{len(skipped_done) - 4})" if len(skipped_done) > 4 else "")
              + "   [--redo to force]\n")
    if not files:
        sys.exit("nothing left to do — every image is already segmented.")

    views, failures, each_ids = [], [], []
    print(f"{len(files)} image(s) from {src}\n")
    print(f"  {'file':<26} {'projection':<12} {'hfov':>6} {'bearing':>8} {'alt':>7}  how")

    for path in files:
        if path.suffix.lower() in (".heic", ".heif") and not HEIC_OK:
            failures.append((path.name, "HEIC support missing — pip install pillow-heif"))
            continue
        print(f"  {path.name:<26} reading …", end="\r", flush=True)
        t0 = time.time()
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            failures.append((path.name, str(e)))
            print(f"  {path.name:<26} FAILED — {e}")
            continue

        v = classify(path, img)
        if args.heading is not None:
            v["heading"] = args.heading % 360
            v["heading_ref"] = "T"
        bearing = f"{v['heading']:.0f}deg" if "heading" in v else "—"
        alt = f"{v['elev_m']:.0f}m" if "elev_m" in v else "—"
        print(f"  {path.name:<26} {v['projection']:<12} {v['hfov_deg']:>5.0f}deg "
              f"{bearing:>8} {alt:>7}  {v['how']}  [{time.time() - t0:.1f}s]")
        if time.time() - t0 > 5:
            print("      ^ that took a while. If resources/ is a symlink into Dropbox, "
                  "iCloud or a network\n        volume, the first read has to download "
                  "the file. Make the folder available\n        offline, or copy the "
                  "panoramas to a local disk first.")

        if v["projection"] == "dual_fisheye":
            failures.append((path.name,
                             "two fisheye circles, not stitched — unusable as geometry. "
                             "Use the Insta360 Studio export instead: the stitched JPEGs "
                             "in resources/panoramas360_static/ carry GPano metadata and "
                             "are detected exactly."))
            continue

        v["source"] = path.name
        v["path"] = path
        v["img"] = img

        # --each: write this scene NOW, then let the image go.
        #
        # Two reasons, and the second is the serious one. Deferring every write to a
        # second loop means nothing appears on disk until the whole run finishes, so an
        # interrupted run leaves nothing. And holding every decoded image until then
        # costs ~215 MB each — thirteen 71-megapixel panoramas is 2.8 GB of RSS, which
        # on a laptop means swapping, which looks exactly like the tool having hung.
        if args.each and not args.catalog:
            scene_id = (f"{args.scene}-{len(each_ids) + 1}" if args.number
                        else slug(path.stem))
            cat = category_dir(v["projection"])
            build_scene(v, Path(args.out) / cat / scene_id, scene_id, args.name)
            each_ids.append((scene_id, cat))
            img.close()
            v.pop("img", None)
        views.append(v)

    if not views:
        print()
        for name, why in failures:
            print(f"  ! {name}: {why}")
        sys.exit("nothing usable")

    # ---------------------------------------------------------------- catalog mode
    if args.catalog:
        clusters = cluster_by_location(views, args.cluster_radius)
        ungeotagged = [v for v in views if "lat" not in v]

        print(f"\n{len(clusters)} candidate point(s) of interest "
              f"within {args.cluster_radius:.0f} m of each other:\n")
        for i, c in enumerate(clusters, 1):
            lat, lon = c["centre"]
            alts = [v["elev_m"] for v in c["views"] if "elev_m" in v]
            widest = max(c["views"], key=lambda v: v["hfov_deg"])
            times = sorted(v["utc"] for v in c["views"] if "utc" in v)
            print(f"  PoI {i}: {lat:.5f}, {lon:.5f}"
                  + (f"  {sum(alts)/len(alts):.0f} m" if alts else "")
                  + f"  — {len(c['views'])} shot(s)")
            print(f"        widest: {widest['source']} at {widest['hfov_deg']:.0f}deg "
                  f"({widest['projection']})")
            if times:
                print(f"        {times[0]}  ->  {times[-1]}")
            print(f"        https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=16/{lat:.5f}/{lon:.5f}")
        if ungeotagged:
            print(f"\n  {len(ungeotagged)} shot(s) without GPS: "
                  + ", ".join(v["source"] for v in ungeotagged[:6])
                  + (" …" if len(ungeotagged) > 6 else ""))

        csv_path = Path(args.out).parent / "photo-catalog.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["source", "projection", "hfov_deg", "vfov_deg", "heading", "heading_ref",
                "lat", "lon", "elev_m", "gps_error_m", "utc", "Model", "how"]
        with csv_path.open("w") as fh:
            fh.write(",".join(cols) + "\n")
            for v in views:
                fh.write(",".join(
                    '"%s"' % str(v.get(c, "")).replace('"', "'") for c in cols) + "\n")
        print(f"\nwrote {rel(csv_path)}")
        print("Pick a PoI, then re-run with --pick <files> --scene <that-poi>.")
        return

    # ---------------------------------------------------------------- one scene each
    if args.each:
        ids = each_ids                       # [(scene_id, category), ...]
        print(f"\n{len(ids)} scene(s) written under {rel(Path(args.out))}")
        _refresh_index(Path(args.out))
        dirs = [f"scenes/{cat}/{i}" for i, cat in ids[:3]]
        print("\nSegment them all:")
        print("  for d in " + " ".join(dirs)
              + (" ..." if len(ids) > 3 else "") + "; do")
        print("    python src/python/image_analysis/segment_panorama.py \\")
        print('        "$d/panorama.jpg" --out "$d" --scene-json "$d/scene.json" --open-vocab')
        print("  done")
        print(f"\nThen give one audio:  python src/python/audio_prep/prepare_audio.py "
              f"--scene {ids[0][0]} --labelled-only")
        if failures:
            print()
            for name, why in failures:
                print(f"  ! {name}: {why}")
        return

    # ---------------------------------------------------------------- build one scene
    # The widest view becomes the scene's main image — that's the one worth panning.
    views.sort(key=lambda v: -v["hfov_deg"])
    main_view = views[0]
    for v in views[1:]:                      # the rest are decoded pixels we never use
        if v.get("img") is not None:
            v["img"].close()
            v["img"] = None
    out_dir = Path(args.out) / category_dir(main_view["projection"]) / args.scene
    build_scene(main_view, out_dir, args.scene, args.name, verbose=True)
    _refresh_index(Path(args.out))
    if failures:
        print()
        for name, why in failures:
            print(f"  ! {name}: {why}")


def build_scene(main_view, out_dir, scene_id, stem, verbose=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    dest, (w, h) = convert(main_view["path"], main_view["img"],
                           main_view["projection"], out_dir, stem)

    panorama = {
        "file": dest.name,
        "projection": main_view["projection"],
        "hfov_deg": round(main_view["hfov_deg"], 1),
        "vfov_deg": round(main_view["vfov_deg"], 1),
        "width": w, "height": h,
        "wrap": main_view["projection"] == "equirect" and main_view["hfov_deg"] > 350,
        "detected_by": main_view["how"],
    }
    for k in ("lat", "lon", "elev_m", "gps_error_m", "utc"):
        if k in main_view:
            panorama[k] = (round(main_view[k], 6) if isinstance(main_view[k], float)
                           else main_view[k])

    panorama["source"] = main_view["source"]

    scene_path = out_dir / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        note = "updated panorama block in"
    else:
        scene = {"id": scene_id, "name": scene_id.replace("-", " ").title(),
                 "north_offset_deg": 0,
                 "scene": {"scenicness": 8.0, "eventfulness": 6.5}, "layers": []}
        note = "created"
    scene["panorama"] = panorama

    # GPSImgDirection is the bearing the camera was pointing, i.e. the bearing of the
    # image CENTRE — which is exactly what north_offset_deg means to the viewer.
    heading = main_view.get("heading")
    if heading is not None:
        scene["north_offset_deg"] = round(float(heading), 1)
        panorama["heading_deg"] = round(float(heading), 1)
        panorama["heading_ref"] = main_view.get("heading_ref", "T")

    scene_path.write_text(json.dumps(scene, indent=2))

    if not verbose:
        head = f"{heading:.0f}deg" if heading is not None else "no bearing"
        print(f"  {scene_id:<16} {main_view['source']:<24} "
              f"{main_view['hfov_deg']:>5.0f}deg  {head:<10} -> {rel(scene_path)}")
        return scene_path

    print(f"\n  main view -> {rel(dest)}  ({w}x{h})")
    print(f"  {note} {rel(scene_path)}")

    if heading is not None:
        ref = main_view.get("heading_ref", "T")
        kind = "true north" if ref == "T" else "magnetic north"
        print(f"  north_offset_deg = {heading:.1f} from EXIF GPSImgDirection ({kind}) —")
        print("  azimuths in this scene are real compass bearings.")
        if main_view["projection"] != "flat":
            print("  For a swept panorama this is the bearing at capture, so treat it as")
            print("  within ~15deg and nudge it if a known landmark sits off-centre.")
    else:
        print("\n  No compass heading in EXIF. Azimuths are relative to the image centre;")
        print("  set north_offset_deg by hand once you know the true bearing.")

    if main_view["projection"] == "flat":
        print("\n  Note: this is a normal photo, so there is only a little room to pan.")
        print("  Prefer a sweep panorama or a 360 for the same spot — the")
        print("  viewport-follows-your-gaze effect needs somewhere to look.")
    if "guessed" in main_view["how"] or "assumed" in main_view["how"]:
        print(f"\n  hfov was guessed ({main_view['hfov_deg']:.0f}deg). If panning feels")
        print("  too fast or slow against the audio, edit panorama.hfov_deg by hand.")
    return scene_path


if __name__ == "__main__":
    main()
