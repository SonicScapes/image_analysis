/**
 * app.js — the viewer.
 *
 * You stand at the point the photograph was taken and turn your head. What you turn
 * towards is what you hear.
 *
 * The picture is re-projected for the direction you are facing (see sphere.js), so the
 * horizon stays straight and verticals stay vertical however far you look up or down.
 * The whole viewer is one small state object:
 *
 *     view = { yaw, pitch, fovH }        degrees, yaw 0 = the centre of the source image
 *
 * `fovH` is the field of view on screen, which is the same thing as zoom but in the unit
 * the audio engine already speaks. Everything else — the drag, the wheel, the compass,
 * the mix — is a function of those three numbers.
 *
 * Bearings: the segmenter measured every region's azimuth with image centre = 0, and
 * `north_offset_deg` says what compass bearing that centre points at. So the engine and
 * the compass get `north_offset_deg + yaw`, while the renderer gets the raw `yaw`.
 *
 *   projection "equirect"     360° — turning wraps around
 *   projection "cylindrical"  iPhone sweep, ~150° — turning stops at the edges
 *   projection "flat"         normal photo, ~50-70° — rendered through its own pinhole
 */

import { SoundscapeEngine } from '../engine/soundscape-engine.js';
import { SphereView } from './sphere.js';

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const fail = (msg) => { const e = $('err'); e.innerHTML = msg; e.style.display = 'block'; };

// Tells the inline boot-check in index.html that the module got this far.
window.__soundscapesReady = true;

/* ------------------------------------------------------------------- routing */
/*
 * A scene is addressable three ways, all of which mean the same thing:
 *
 *   /scene/IMG_20260912_151140_00_029          pretty (serve.json rewrites it here)
 *   .../index.html#/scene/IMG_…_029            works with any static server
 *   .../index.html?scene=…/scene.json          a scene.json outside scenes/
 *
 * The address bar is kept in step as you move between scenes, so whatever is on screen
 * can be copied and sent to someone.
 */

const params = new URLSearchParams(location.search);
const SCENES_ROOT = new URL('../../../scenes/', document.baseURI).href;

function routeId() {
  const q = params.get('scene');
  if (q && !q.includes('/')) return q;
  const h = location.hash.replace(/^#\/?(scene\/)?/, '');
  if (h) return decodeURIComponent(h);
  const p = location.pathname.match(/\/scene\/([^/?#]+)/);
  return p ? decodeURIComponent(p[1]) : null;
}

/** A scene.json somewhere else entirely, for trying one out before it is in the index. */
const EXPLICIT_URL = (() => {
  const q = params.get('scene');
  return q && q.includes('/') ? new URL(q, document.baseURI).href : null;
})();

function writeUrl(id) {
  if (!id) return;
  // Keep whichever form the visitor arrived in: rewriting /scene/x into #/scene/x (or
  // the reverse) under someone's feet makes the back button behave oddly.
  const pretty = /\/scene\/[^/?#]+/.test(location.pathname);
  const next = pretty
    ? location.pathname.replace(/\/scene\/[^/?#]+/, '/scene/' + encodeURIComponent(id))
    : location.pathname + location.search + '#/scene/' + encodeURIComponent(id);
  try { history.replaceState(null, '', next); } catch { /* file://, not important */ }
}

/* --------------------------------------------------------------------- state */

const stage = $('stage');
const engine = new SoundscapeEngine();

let scene = null;
let pano = null;
let sceneBase = SCENES_ROOT;
let sceneList = [];
let sceneIdx = 0;
let silentMode = false;

let img = { w: 2, h: 1 };
let renderer = null;
let geom = { hfov: 360, vfov: 180, wrap: true, flat: false };

/** Degrees. yaw 0 is the centre of the source image; pitch 0 is the horizon. */
const view = { yaw: 0, pitch: 0, fovH: 55 };

window.__soundscapes = {
  engine, view,
  get scene() { return scene; },
  get sceneList() { return sceneList; },
  look: (yaw, pitch = 0) => { view.yaw = yaw; view.pitch = pitch; clampView(); },
};

/* ------------------------------------------------------------------ geometry */

/** Vertical field of view that matches the window's shape at the current fovH. */
function fovV() {
  const a = Math.max(stage.clientHeight, 1) / Math.max(stage.clientWidth, 1);
  return 2 * Math.atan(Math.tan((view.fovH * Math.PI) / 360) * a) * 180 / Math.PI;
}

/**
 * A human looks at about 75 degrees at a time. We open a little tighter than that so
 * the first impression is standing in the place, not looking at a wide still.
 *
 * Opening a 360 at its full width would show 360 degrees at once — a picture OF a
 * panorama rather than standing in one — and it also flattens the sound: with everything
 * visible, nothing is ever out of view, so the mix barely moves as you turn.
 */
const TARGET_FOV = 55;
function resetView() {
  view.fovH = Math.min(TARGET_FOV, geom.hfov);
  view.yaw = 0;
  view.pitch = 0;
  clampView();
}

function clampView() {
  view.fovH = clamp(view.fovH, 12, Math.min(120, geom.hfov));
  const v = fovV();
  if (geom.wrap) {
    view.yaw = ((view.yaw % 360) + 540) % 360 - 180;      // -180..180, no edges
  } else {
    view.yaw = clamp(view.yaw, -Math.max(0, (geom.hfov - view.fovH) / 2),
                               Math.max(0, (geom.hfov - view.fovH) / 2));
  }
  const room = geom.wrap ? 90 : Math.max(0, (geom.vfov - v) / 2);
  view.pitch = clamp(view.pitch, -room, room);
}

/** What the engine and the compass need: an absolute compass bearing. */
function bearing() {
  return (((scene?.north_offset_deg ?? 0) + view.yaw) % 360 + 360) % 360;
}

/* --------------------------------------------------------------------- input */
/*
 * Degrees per pixel is taken at the centre of the screen, where tan() is locally linear.
 * Using fovH/width instead would feel sluggish when zoomed out, because a wide
 * perspective view covers more angle per pixel at the edges than in the middle.
 */
function degPerPx() {
  const R = 180 / Math.PI;
  return {
    x: (2 * Math.tan((view.fovH * Math.PI) / 360) / Math.max(stage.clientWidth, 1)) * R,
    y: (2 * Math.tan((fovV() * Math.PI) / 360) / Math.max(stage.clientHeight, 1)) * R,
  };
}

const pointers = new Map();
let pinch = null;

stage.addEventListener('pointerdown', (e) => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  stage.setPointerCapture(e.pointerId);
  stage.classList.add('dragging');
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), fov: view.fovH };
  }
});

stage.addEventListener('pointermove', (e) => {
  const prev = pointers.get(e.pointerId);
  if (!prev || useGyro) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (pinch && pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    if (d > 4) view.fovH = pinch.fov * (pinch.dist / d);
    clampView();
    return;
  }
  const k = degPerPx();
  view.yaw -= (e.clientX - prev.x) * k.x;     // drag right, look left
  view.pitch += (e.clientY - prev.y) * k.y;   // drag down, look up
  clampView();
});

const release = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
  if (!pointers.size) stage.classList.remove('dragging');
};
stage.addEventListener('pointerup', release);
stage.addEventListener('pointercancel', release);

stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  view.fovH *= e.deltaY > 0 ? 1.08 : 0.92;
  clampView();
}, { passive: false });

addEventListener('resize', clampView);

let useGyro = false;
if (typeof DeviceOrientationEvent !== 'undefined' && 'ontouchstart' in window) {
  $('gyroBtn').hidden = false;
  $('gyroBtn').addEventListener('click', async () => {
    if (DeviceOrientationEvent.requestPermission) {
      const ok = await DeviceOrientationEvent.requestPermission().catch(() => 'denied');
      if (ok !== 'granted') return fail('Device motion denied — drag to look around instead.');
    }
    useGyro = true;
    $('gyroBtn').textContent = 'Device motion on';
    // The compass gives an absolute bearing; the renderer wants it relative to the
    // image centre, which is exactly what north_offset_deg converts between.
    addEventListener('deviceorientation', (e) => {
      if (e.alpha == null) return;
      const heading = (360 - e.alpha) % 360;
      view.yaw = (((heading - (scene?.north_offset_deg ?? 0)) % 360) + 540) % 360 - 180;
      view.pitch = clamp((e.beta ?? 90) - 90, -80, 80);
      clampView();
    });
  });
}

addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const step = view.fovH / 6;
  if (e.key === 'a' || e.key === 'A') { view.yaw -= step; clampView(); }
  if (e.key === 'd' || e.key === 'D') { view.yaw += step; clampView(); }
  if (e.key === 'w' || e.key === 'W') { view.pitch += step / 2; clampView(); }
  if (e.key === 's' || e.key === 'S') { view.pitch -= step / 2; clampView(); }
  if (e.key === '+' || e.key === '=') { view.fovH *= 0.9; clampView(); }
  if (e.key === '-') { view.fovH *= 1.1; clampView(); }
  if (e.key === 'ArrowLeft' || e.key === '[') { e.preventDefault(); gotoScene(-1); }
  if (e.key === 'ArrowRight' || e.key === ']') { e.preventDefault(); gotoScene(1); }
  const n = ['srcPhoto', 'srcOverlay', 'srcNamed', 'srcLabels'][+e.key - 1];
  if (n) setSource(n);
});

/* ------------------------------------------------------------------ renderer */

/**
 * If WebGL is unavailable we still show the panorama, just without the reprojection:
 * a CSS background offset, the way the viewer worked before. The horizon bows and
 * looking up smears, but the demo runs and the sound is unaffected.
 */
class FlatFallback {
  constructor(host) {
    this.el = host;
    this.el.style.backgroundRepeat = 'repeat-x';
    this.degraded = true;
  }
  setImage(image, geometry) {
    this.el.style.backgroundImage = `url("${image.src}")`;
    this.g = geometry;
    this.aspect = (image.naturalHeight || 1) / (image.naturalWidth || 1);
  }
  render({ yaw, pitch, fovH, fovV: fv }) {
    const W = this.el.clientWidth * (this.g.hfov / fovH);
    const H = W * this.aspect;
    this.el.style.backgroundSize = `${W}px ${H}px`;
    const x = (yaw / this.g.hfov + 0.5) * W - this.el.clientWidth / 2;
    const y = (0.5 - pitch / this.g.vfov) * H - this.el.clientHeight / 2;
    this.el.style.backgroundPosition = `${-x}px ${-y}px`;
  }
}

function makeRenderer() {
  try {
    const r = new SphereView($('gl'));
    $('gl').hidden = false;
    return r;
  } catch (err) {
    console.warn('[soundscapes] WebGL unavailable, falling back to flat pan:', err.message);
    $('gl').hidden = true;
    $('flat').hidden = false;
    fail('<b>WebGL unavailable — flat panning</b><br>The picture will bow at the edges. ' +
         'Everything else, including the sound, works normally.');
    return new FlatFallback($('flat'));
  }
}

/* --------------------------------------------------------------------- start */

async function boot(silent) {
  silentMode = silent;
  $('startBtn').disabled = true;
  $('exploreBtn').disabled = true;
  try {
    renderer = renderer || makeRenderer();
    try {
      const res = await fetch(SCENES_ROOT + 'index.json', { cache: 'no-store' });
      if (res.ok) sceneList = (await res.json()).scenes ?? [];
    } catch { /* no index: fall back to ?scene= */ }

    const wanted = routeId();
    sceneIdx = Math.max(0, sceneList.findIndex((s) => s.id === wanted));
    if (sceneList.length) {
      $('sceneNav').hidden = false;
      $('sceneSelect').innerHTML = sceneList.map((s, i) =>
        `<option value="${i}">${i + 1}/${sceneList.length}  ${s.name}` +
        `${s.layers ? '' : '  (no audio yet)'}</option>`).join('');
    } else if (!EXPLICIT_URL) {
      throw new Error('no scenes found — run ingest_images.py, then build_scene_layers.py');
    }

    if (!looping) { looping = true; frame(); }
    await openScene(sceneIdx);
    $('silentBadge').hidden = !silent;
    $('gate').style.display = 'none';
  } catch (err) {
    $('startBtn').disabled = false;
    $('exploreBtn').disabled = false;
    $('status').textContent = err.message;
    const hint = /could not load|Failed to fetch|NetworkError/.test(err.message)
      ? 'Serve the repo ROOT over http (npm run dev), not the app folder, and not file://'
      : 'Check the console (Cmd+Opt+J) for the full trace.';
    fail(`<b>${err.message}</b><br>${hint}`);
    console.error('[soundscapes]', err);
  }
}

async function openScene(i) {
  const entry = sceneList[i];
  // `dir` is the scene's path relative to the scenes root, including its category
  // subfolder (360pano/<id>, photos/<id>) — `id` alone is just the folder's own name,
  // which index.json also carries for display and for matching a routed /scene/<id>.
  const base = entry ? SCENES_ROOT + entry.dir + '/'
                     : EXPLICIT_URL.replace(/[^/]+$/, '');
  const url = entry ? base + 'scene.json' : EXPLICIT_URL;

  $('status').textContent = 'loading…';
  const next = await (await fetch(url)).json();
  if (!next.panorama) throw new Error('scene.json has no "panorama" block — run ingest_images.py');
  next.panorama.vfov_deg = next.panorama.vfov_deg ?? next.panorama.hfov_deg / 2;

  if (engine.running) await engine.stop(0.25);
  scene = next;
  pano = next.panorama;
  sceneIdx = i;
  sceneBase = base;

  await loadImage(base + pano.file, 'srcPhoto');
  resetView();

  // A scene with no layers is normal while the audio is still being built — open it and
  // say so, rather than refusing to show the panorama.
  const layers = scene.layers ?? [];
  $('noAudio').hidden = layers.length > 0;
  await engine.loadScene({ ...scene, layers }, {
    baseUrl: base,
    silent: silentMode || layers.length === 0,
    onProgress: (d, t) => { $('status').textContent = `loading sounds… ${d}/${t}`; },
  });
  await engine.start();

  $('poiName').textContent = scene.name ?? scene.id;
  $('scenic').value = scene.scene?.scenicness ?? 8;
  $('scenicVal').textContent = (+$('scenic').value).toFixed(1);
  $('pressure').value = scene.scene?.pressure ?? 0;
  $('pressureVal').textContent = (+$('pressure').value).toFixed(1);
  if (sceneList.length) $('sceneSelect').value = String(i);
  buildRows();
  for (const el of ['title', 'meters', 'compass', 'controls']) $(el).hidden = false;
  currentSource = 'srcPhoto';
  for (const k of Object.keys(SOURCES)) $(k).setAttribute('aria-pressed', String(k === 'srcPhoto'));
  $('srcNote').hidden = true;
  writeUrl(entry?.id);
  showWhere();
  await detectSources();
}

async function gotoScene(delta) {
  if (!sceneList.length) return;
  const i = (sceneIdx + delta + sceneList.length) % sceneList.length;
  try {
    await openScene(i);
  } catch (err) {
    fail(`${sceneList[i]?.id}: ${err.message}`);
  }
}

$('prevScene').addEventListener('click', () => gotoScene(-1));
$('nextScene').addEventListener('click', () => gotoScene(1));
$('sceneSelect').addEventListener('change', (e) => {
  openScene(+e.target.value).catch((err) => fail(err.message));
});
$('startBtn').addEventListener('click', () => boot(false));
$('exploreBtn').addEventListener('click', () => boot(true));

/* --------------------------------------------------------------- where we are */

/**
 * Latitude, longitude, altitude and — where we can get one — the name of the place.
 *
 * The name is baked into scene.json by geocode_scenes.py so the demo needs no network.
 * If it is missing we ask OpenStreetMap once and remember the answer in localStorage,
 * which keeps a laptop that is online useful without making the presentation depend on
 * the venue's wifi.
 */
function hasGps(p) {
  return p && Number.isFinite(p.lat) && Number.isFinite(p.lon)
      && (Math.abs(p.lat) > 1e-4 || Math.abs(p.lon) > 1e-4);
}

function formatGps(p) {
  const ns = p.lat >= 0 ? 'N' : 'S', ew = p.lon >= 0 ? 'E' : 'W';
  const alt = Number.isFinite(p.elev_m) && Math.abs(p.elev_m) > 1
    ? ` · ${Math.round(p.elev_m)} m` : '';
  return `${Math.abs(p.lat).toFixed(5)}° ${ns}  ${Math.abs(p.lon).toFixed(5)}° ${ew}${alt}`;
}

async function placeName(lat, lon) {
  const key = `ss:place:${lat.toFixed(4)},${lon.toFixed(4)}`;
  try { const hit = localStorage.getItem(key); if (hit) return hit; } catch { /* private mode */ }
  const u = 'https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=14'
          + `&lat=${lat}&lon=${lon}`;
  const j = await (await fetch(u, { headers: { Accept: 'application/json' } })).json();
  const a = j.address ?? {};
  const name = [a.hamlet || a.village || a.town || a.suburb || a.locality || j.name,
                a.municipality || a.city || a.county,
                a.state].filter(Boolean).slice(0, 2).join(', ');
  if (!name) throw new Error('no name for this point');
  try { localStorage.setItem(key, name); } catch { /* fine */ }
  return name;
}

function showWhere() {
  const box = $('geo'), nameEl = $('placeName');
  if (!hasGps(pano)) { box.hidden = true; nameEl.textContent = ''; return; }
  box.hidden = false;
  box.textContent = formatGps(pano);
  const baked = scene.place || pano.place;
  if (baked) { nameEl.textContent = baked; return; }
  nameEl.textContent = '';
  const want = sceneIdx;
  placeName(pano.lat, pano.lon)
    .then((n) => { if (sceneIdx === want) nameEl.textContent = n; })
    .catch(() => { /* offline, or OSM has nothing here — the coordinates still show */ });
}

/* ------------------------------------------------- displayed image switcher */

/*
 * overlay.png and overlay_labeled.png are transparent tints — mostly empty alpha, opaque
 * only over classified ground — not full copies of the photo. That is what keeps them
 * small (a flat-coloured, mostly-transparent PNG compresses to a fraction of a percent
 * of a photographic one). So showing them means drawing the photo, then the tint on top,
 * not swapping the photo out for the tint. `composite: true` marks the two that need that.
 */
const SOURCES = {
  srcPhoto:   { file: () => pano.file, label: 'the photograph' },
  srcOverlay: { file: () => 'overlay.png', label: 'segmentation painted over the photo', composite: true },
  srcNamed:   { file: () => 'overlay_labeled.png', label: 'overlay with class names', composite: true },
  srcLabels:  { file: () => 'labels.png', label: 'the raw sphere label map' },
};
let currentSource = 'srcPhoto';

const INDEX_FLAG = { srcOverlay: 'overlay', srcNamed: 'overlay_labeled', srcLabels: 'labels' };

async function detectSources() {
  const entry = sceneList[sceneIdx];
  for (const id of Object.keys(SOURCES)) {
    if (id === 'srcPhoto') continue;
    let ok;
    if (entry && INDEX_FLAG[id] in entry) {
      ok = !!entry[INDEX_FLAG[id]];          // the index already knows
    } else {
      try {
        ok = (await fetch(sceneBase + SOURCES[id].file(), { method: 'HEAD' })).ok;
      } catch { ok = false; }
    }
    $(id).disabled = !ok;
    $(id).title = ok ? SOURCES[id].label : 'not generated yet — run segment_panorama.py';
  }
}

/**
 * Which sphere a given image covers.
 *
 * overlay.png and overlay_labeled.png are painted on the source image, so they share its
 * geometry. labels.png is always a 2:1 map of the whole sphere — over a flat photo or a
 * partial sweep that is a different shape from the picture, so we say so rather than let
 * it look subtly misaligned.
 */
function geometryFor(id) {
  const g = {
    hfov: pano.hfov_deg, vfov: pano.vfov_deg,
    wrap: !!pano.wrap, flat: pano.projection === 'flat',
  };
  if (id === 'srcPhoto') return g;
  const aspect = img.w / img.h;
  if (Math.abs(aspect - 2) < 0.06 && pano.hfov_deg < 355) {
    return { hfov: 360, vfov: 180, wrap: true, flat: false, sphere: true };
  }
  return g;
}

async function setSource(id) {
  if ($(id).disabled || id === currentSource) return;
  try {
    const spec = SOURCES[id];
    if (spec.composite) {
      await loadComposite(sceneBase + pano.file, sceneBase + spec.file(), id);
    } else {
      await loadImage(sceneBase + spec.file(), id);
    }
  } catch (err) {
    fail(`Could not load ${SOURCES[id].file()} — ${err.message}`);
    return;
  }
  currentSource = id;
  for (const k of Object.keys(SOURCES)) $(k).setAttribute('aria-pressed', String(k === id));
  const note = $('srcNote');
  note.hidden = !geom.sphere;
  if (geom.sphere) note.textContent = 'full sphere — wider than this photograph';
  clampView();
}

/** Draw the photo, then a transparent tint on top, and hand the result to the renderer
 * as one image — the tint alone would just be a mostly-black sparse patchwork. */
function loadComposite(baseUrl, tintUrl, sourceId) {
  return new Promise((resolve, reject) => {
    const base = new Image();
    base.onload = () => {
      const tint = new Image();
      tint.onload = () => {
        const c = document.createElement('canvas');
        c.width = base.naturalWidth;
        c.height = base.naturalHeight;
        const ctx = c.getContext('2d');
        ctx.drawImage(base, 0, 0, c.width, c.height);
        ctx.drawImage(tint, 0, 0, c.width, c.height);
        const merged = new Image();
        merged.onload = () => {
          img = { w: merged.naturalWidth, h: merged.naturalHeight };
          geom = geometryFor(sourceId);
          renderer.setImage(merged, geom);
          renderer.render({ yaw: view.yaw, pitch: view.pitch, fovH: view.fovH, fovV: fovV() });
          resolve();
        };
        merged.onerror = () => reject(new Error('could not composite the tint'));
        merged.src = c.toDataURL('image/png');
      };
      tint.onerror = () => reject(new Error(`could not load ${tintUrl}`));
      tint.src = tintUrl;
    };
    base.onerror = () => reject(new Error(`could not load ${baseUrl}`));
    base.src = baseUrl;
  });
}

for (const id of Object.keys(SOURCES)) {
  $(id).addEventListener('click', () => setSource(id));
}

function loadImage(url, sourceId) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => {
      img = { w: el.naturalWidth, h: el.naturalHeight };
      geom = geometryFor(sourceId);
      renderer.setImage(el, geom);
      renderer.render({ yaw: view.yaw, pitch: view.pitch, fovH: view.fovH, fovV: fovV() });
      resolve();
    };
    el.onerror = () => reject(new Error(`could not load ${url}`));
    el.src = url;
  });
}

/* ------------------------------------------------------------------ controls */

$('scenic').addEventListener('input', (e) => {
  $('scenicVal').textContent = (+e.target.value).toFixed(1);
  engine.setMood({ scenicness: +e.target.value });
});
$('pressure').addEventListener('input', (e) => {
  $('pressureVal').textContent = (+e.target.value).toFixed(1);
  engine.setPressure(+e.target.value);
});

/* -------------------------------------------------------------------- meters */

const rowEls = new Map();
function buildRows() {
  const host = $('rows');
  host.innerHTML = '';
  rowEls.clear();
  for (const l of engine.getDebug()) {
    const wrap = document.createElement('div');
    wrap.className = 'mrow ' + l.type;
    const files = l.files.length > 1 ? `${l.files[0]} +${l.files.length - 1}` : (l.files[0] ?? '—');
    wrap.innerHTML =
      `<div class="row ${l.type}"><span class="lbl">${l.id}</span>` +
      `<span class="bar"><i></i></span><span class="db">—</span></div>` +
      `<div class="sub"><span class="file" title="${l.files.join(', ')}">${files}</span>` +
      `<span class="pct">—</span></div>`;
    host.appendChild(wrap);
    rowEls.set(l.id, {
      row: wrap.querySelector('.row'), wrap,
      fill: wrap.querySelector('i'), db: wrap.querySelector('.db'),
      pct: wrap.querySelector('.pct'),
    });
  }
}

const dots = $('dots');
function drawCompass(debug, v) {
  const a0 = (v.yaw - v.fovH / 2 - 90) * Math.PI / 180;
  const a1 = (v.yaw + v.fovH / 2 - 90) * Math.PI / 180;
  const p = (a, r) => `${(66 + Math.cos(a) * r).toFixed(1)},${(66 + Math.sin(a) * r).toFixed(1)}`;
  $('wedge').setAttribute('d',
    `M66,66 L${p(a0, 52)} A52,52 0 ${v.fovH > 180 ? 1 : 0},1 ${p(a1, 52)} Z`);

  let svg = '';
  for (const l of debug) {
    if (l.type === 'bed' || l.type === 'score') continue;
    const a = (l.az - 90) * Math.PI / 180;
    const r = 20 + Math.min(l.distance, 500) / 500 * 30;
    const fresh = engine.now() - l.lastEventAt < 0.6;
    const col = l.type === 'event' ? '#E09257' : '#58B9BE';
    const rad = 2 + l.visibility * 3.2 + (fresh ? 3 : 0);
    svg += `<circle cx="${(66 + Math.cos(a) * r).toFixed(1)}" cy="${(66 + Math.sin(a) * r).toFixed(1)}"` +
           ` r="${rad.toFixed(1)}" fill="${col}" opacity="${(0.3 + l.visibility * 0.7).toFixed(2)}"/>`;
  }
  dots.innerHTML = svg;
}

let looping = false;
function frame() {
  requestAnimationFrame(frame);
  const v = { yaw: view.yaw, pitch: view.pitch, fovH: view.fovH, fovV: fovV() };
  if (renderer) renderer.render(v);
  if (!engine.running) return;

  // The engine thinks in compass bearings; the renderer thinks in image coordinates.
  const heard = { ...v, yaw: bearing() };
  engine.setView(heard);

  const debug = engine.getDebug();
  // Share of the mix: power, normalised across whatever is currently audible. An event
  // only counts while it is sounding, which is why its share blinks.
  const totalPower = debug.reduce((s, l) => s + l.power, 0) || 1;
  for (const l of debug) {
    const el = rowEls.get(l.id);
    if (!el) continue;
    el.fill.style.width = (l.visibility * 100).toFixed(0) + '%';
    el.db.textContent = l.type === 'event'
      ? (l.visibility * 100).toFixed(0) + '%'
      : l.gainDb.toFixed(0);
    const share = (l.power / totalPower) * 100;
    el.pct.textContent = l.power > 0
      ? (share < 1 ? '<1%' : share.toFixed(0) + '%')
      : (l.type === 'event' ? 'idle' : '—');
    el.row.classList.toggle('fired', engine.now() - l.lastEventAt < 0.5);
  }
  drawCompass(debug, heard);
  $('bearing').textContent = Math.round(heard.yaw) + '°';
  $('fovOut').textContent = Math.round(v.fovH) + '°';
}
