/**
 * app.js — the viewer.
 *
 * Renders whatever `ingest_images.py` produced and turns panning into a bearing, then
 * hands that bearing to the audio engine. Deliberately 2D: the image is a window onto a
 * sphere with a known horizontal field of view, so a CSS transform is all the geometry
 * we need. No WebGL, no library, nothing to fail on stage.
 *
 *   projection "equirect"     360° — panning wraps around
 *   projection "cylindrical"  iPhone sweep, ~150° — panning stops at the edges
 *   projection "flat"         normal photo, ~50-70° — zoom in and there is still room to look
 *
 * Same code path for all three. Only `hfov_deg` differs.
 */

import { SoundscapeEngine } from '../engine/soundscape-engine.js';

const params = new URLSearchParams(location.search);
const SCENE_URL = params.get('scene') || '/data/scenes/hohe-tauern/scene.json';
const BASE_URL = SCENE_URL.replace(/[^/]+$/, '');

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const fail = (msg) => { const e = $('err'); e.innerHTML = msg; e.style.display = 'block'; };

// Tells the inline boot-check in index.html that the module got this far.
window.__soundscapesReady = true;

const stage = $('stage');
const strip = $('strip');
const engine = new SoundscapeEngine();

// Handy in the console: __soundscapes.engine.getDebug(), .engine.setMood({...}), .scene
window.__soundscapes = { engine, get scene() { return scene; },
                         get sceneList() { return sceneList; } };

let scene = null;
let pano = null;
let sceneBase = BASE_URL;
let img = { w: 1, h: 1 };
let pan = { x: 0, y: 0 };
let zoom = 1;
let dragging = false, last = { x: 0, y: 0 }, useGyro = false;

/* ------------------------------------------------------------------ geometry */

/** Scale at which the image covers the stage, times the zoom factor. */
function baseScale() {
  return Math.max(stage.clientHeight / img.h, stage.clientWidth / img.w);
}

function coverScale() {
  return baseScale() * zoom;
}

/**
 * Choose the zoom that gives a human field of view.
 *
 * Without this a 360 opens at zoom 1, which fits the whole equirect across the window —
 * 288 degrees at once. That is a picture OF a panorama, not standing in one, and it also
 * flattens the audio: with everything visible at once, nothing is ever out of view, so
 * the mix barely moves as you pan. Around 75 degrees is roughly what a person sees.
 */
const TARGET_FOV = 75;
function fitZoom(target = TARGET_FOV) {
  if (!pano || pano.hfov_deg <= target) { zoom = 1; return; }
  const need = (pano.hfov_deg * stage.clientWidth) / (target * img.w);
  zoom = clamp(need / baseScale(), 1, 24);
}

function maxPan() {
  const s = coverScale();
  return {
    x: Math.max(0, img.w * s - stage.clientWidth),
    y: Math.max(0, img.h * s - stage.clientHeight),
  };
}

/** Pan offset (in rendered pixels) -> bearing, pitch and the field of view on screen. */
function viewFromPan() {
  const s = coverScale();
  const rw = img.w * s, rh = img.h * s;
  const fovH = pano.hfov_deg * Math.min(1, stage.clientWidth / rw);
  const fovV = pano.vfov_deg * Math.min(1, stage.clientHeight / rh);

  const cx = (pan.x + stage.clientWidth / 2) / rw;   // 0..1 across the image
  const cy = (pan.y + stage.clientHeight / 2) / rh;
  const yawRel = pano.hfov_deg * (cx - 0.5);
  const pitch = -pano.vfov_deg * (cy - 0.5);

  const north = scene.north_offset_deg ?? 0;
  return { yaw: (north + yawRel + 720) % 360, pitch, fovH, fovV };
}

function applyTransform() {
  const s = coverScale();
  const rw = img.w * s;
  if (pano.wrap) {
    pan.x = ((pan.x % rw) + rw) % rw;
  } else {
    const m = maxPan();
    pan.x = clamp(pan.x, 0, m.x);
  }
  pan.y = clamp(pan.y, 0, maxPan().y);
  strip.style.transform = `translate3d(${-pan.x}px, ${-pan.y}px, 0)`;
  for (const el of strip.children) el.style.height = `${img.h * s}px`;
}

/* --------------------------------------------------------------------- input */

stage.addEventListener('pointerdown', (e) => {
  dragging = true; last = { x: e.clientX, y: e.clientY };
  stage.classList.add('dragging');
  stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointermove', (e) => {
  if (!dragging || useGyro) return;
  pan.x -= e.clientX - last.x;
  pan.y -= e.clientY - last.y;
  last = { x: e.clientX, y: e.clientY };
  applyTransform();
});
const endDrag = () => { dragging = false; stage.classList.remove('dragging'); };
stage.addEventListener('pointerup', endDrag);
stage.addEventListener('pointercancel', endDrag);

stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  const before = coverScale();
  zoom = clamp(zoom * (e.deltaY > 0 ? 0.92 : 1.08), 1, 24);
  // Keep the point under the cursor fixed while zooming.
  const k = coverScale() / before;
  pan.x = (pan.x + e.clientX) * k - e.clientX;
  pan.y = (pan.y + e.clientY) * k - e.clientY;
  applyTransform();
}, { passive: false });

addEventListener('resize', () => { fitZoom(); applyTransform(); });

if (typeof DeviceOrientationEvent !== 'undefined' && 'ontouchstart' in window) {
  $('gyroBtn').hidden = false;
  $('gyroBtn').addEventListener('click', async () => {
    if (DeviceOrientationEvent.requestPermission) {
      const ok = await DeviceOrientationEvent.requestPermission().catch(() => 'denied');
      if (ok !== 'granted') return fail('Device motion denied — drag to look around instead.');
    }
    useGyro = true;
    $('gyroBtn').textContent = 'Device motion on';
    addEventListener('deviceorientation', (e) => {
      if (e.alpha == null) return;
      const s = coverScale(), rw = img.w * s;
      // Map the compass reading back onto a pan offset.
      const rel = (((360 - e.alpha) - (scene.north_offset_deg ?? 0) + 540) % 360) - 180;
      pan.x = (rel / pano.hfov_deg + 0.5) * rw - stage.clientWidth / 2;
      const tilt = clamp((e.beta ?? 90) - 90, -60, 60);
      pan.y = (-tilt / pano.vfov_deg + 0.5) * img.h * s - stage.clientHeight / 2;
      applyTransform();
    });
  });
}

/* --------------------------------------------------------------------- start */

/**
 * Boot the scene. `silent` skips all audio: no AudioContext, no downloads, no graph —
 * but geometry, visibility, meters and compass all still run, which is what you want
 * when checking whether a region is pointing where you think it is.
 */
let sceneList = [];
let sceneIdx = 0;
let silentMode = false;

/**
 * Boot: read the scene index, then open one. The index exists because a browser cannot
 * list a directory over HTTP; `scene_index.py` writes it and every tool that creates a
 * scene refreshes it.
 */
async function boot(silent) {
  silentMode = silent;
  $('startBtn').disabled = true;
  $('exploreBtn').disabled = true;
  try {
    try {
      const res = await fetch(new URL('../', new URL(BASE_URL, location.href)).href + 'index.json');
      if (res.ok) sceneList = (await res.json()).scenes ?? [];
    } catch { /* no index: fall back to the single scene in the URL */ }

    const wanted = SCENE_URL.split('/').slice(-2, -1)[0];
    sceneIdx = Math.max(0, sceneList.findIndex((s) => s.id === wanted));
    if (sceneList.length) {
      $('sceneNav').hidden = false;
      $('sceneSelect').innerHTML = sceneList.map((s, i) =>
        `<option value="${i}">${i + 1}/${sceneList.length}  ${s.name}` +
        `${s.layers ? '' : '  (no audio)'}</option>`).join('');
    }

    await openScene(sceneIdx);
    $('silentBadge').hidden = !silent;
    $('gate').style.display = 'none';
    frame();
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

/** Open scene `i` from the index, or the URL's scene when there is no index. */
async function openScene(i) {
  const entry = sceneList[i];
  const base = entry
    ? new URL(`../${entry.id}/`, new URL(BASE_URL, location.href)).href
    : new URL(BASE_URL, location.href).href;
  const url = base + 'scene.json';

  $('status').textContent = 'loading…';
  const next = await (await fetch(url)).json();
  if (!next.panorama) throw new Error('scene.json has no "panorama" block — run ingest_images.py');
  next.panorama.vfov_deg = next.panorama.vfov_deg ?? next.panorama.hfov_deg / 2;

  if (engine.running) await engine.stop(0.25);
  scene = next;
  pano = next.panorama;
  sceneIdx = i;
  sceneBase = base;

  strip.innerHTML = '';
  await loadImage(base + pano.file);

  // A scene with no layers is normal while audio is still being built — open it anyway
  // and say so, rather than refusing to show the panorama.
  const layers = scene.layers ?? [];
  $('noAudio').hidden = layers.length > 0;
  await engine.loadScene({ ...scene, layers }, {
    baseUrl: base,
    silent: silentMode || layers.length === 0,
    onProgress: (d, t) => { $('status').textContent = `loading sounds… ${d}/${t}`; },
  });
  await engine.start();

  fitZoom();
  const m = maxPan();
  pan = { x: m.x / 2, y: m.y / 2 };
  applyTransform();

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

addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft' || e.key === '[') { e.preventDefault(); gotoScene(-1); }
  if (e.key === 'ArrowRight' || e.key === ']') { e.preventDefault(); gotoScene(1); }
  const n = ['srcPhoto', 'srcOverlay', 'srcNamed', 'srcLabels'][+e.key - 1];
  if (n) setSource(n);
});

$('startBtn').addEventListener('click', () => boot(false));
$('exploreBtn').addEventListener('click', () => boot(true));

/* ------------------------------------------------- displayed image switcher */

const SOURCES = {
  srcPhoto:   { file: () => pano.file, label: 'the photograph' },
  srcOverlay: { file: () => 'overlay.png', label: 'segmentation painted over the photo' },
  srcNamed:   { file: () => 'overlay_labeled.png', label: 'overlay with class names' },
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

async function setSource(id) {
  if ($(id).disabled || id === currentSource) return;
  const prevAspect = img.w / img.h;
  try {
    strip.innerHTML = '';
    await loadImage(sceneBase + SOURCES[id].file());
  } catch (err) {
    fail(`Could not load ${SOURCES[id].file()} — ${err.message}`);
    return;
  }
  currentSource = id;
  for (const k of Object.keys(SOURCES)) $(k).setAttribute('aria-pressed', String(k === id));

  // A label map is a full sphere at 2:1. Over a flat photo or a partial sweep that is a
  // different shape from the picture, so say so rather than letting it look misaligned.
  const note = $('srcNote');
  const aspect = img.w / img.h;
  if (id !== 'srcPhoto' && Math.abs(aspect - prevAspect) > 0.05) {
    note.textContent = 'full sphere — does not line up with this photo';
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

for (const id of Object.keys(SOURCES)) {
  $(id).addEventListener('click', () => setSource(id));
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => {
      img = { w: el.naturalWidth, h: el.naturalHeight };
      strip.appendChild(el);
      // A 360 needs a second copy so the pan can wrap without a gap.
      if (pano.wrap) {
        const clone = el.cloneNode();
        strip.appendChild(clone);
      }
      applyTransform();
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
function drawCompass(debug, view) {
  const a0 = (view.yaw - view.fovH / 2 - 90) * Math.PI / 180;
  const a1 = (view.yaw + view.fovH / 2 - 90) * Math.PI / 180;
  const p = (a, r) => `${(66 + Math.cos(a) * r).toFixed(1)},${(66 + Math.sin(a) * r).toFixed(1)}`;
  $('wedge').setAttribute('d',
    `M66,66 L${p(a0, 52)} A52,52 0 ${view.fovH > 180 ? 1 : 0},1 ${p(a1, 52)} Z`);

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

function frame() {
  requestAnimationFrame(frame);
  if (!engine.running) return;

  const view = viewFromPan();
  engine.setView(view);

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
  drawCompass(debug, view);
  $('bearing').textContent = Math.round(view.yaw) + '°';
  $('fovOut').textContent = Math.round(view.fovH) + '°';
}
