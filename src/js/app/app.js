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

const stage = $('stage');
const strip = $('strip');
const engine = new SoundscapeEngine();

let scene = null;
let pano = null;
let img = { w: 1, h: 1 };
let pan = { x: 0, y: 0 };
let zoom = 1;
let dragging = false, last = { x: 0, y: 0 }, useGyro = false;

/* ------------------------------------------------------------------ geometry */

/** Scale at which the image covers the stage, times the zoom factor. */
function coverScale() {
  return Math.max(stage.clientHeight / img.h, stage.clientWidth / img.w) * zoom;
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
  zoom = clamp(zoom * (e.deltaY > 0 ? 0.92 : 1.08), 1, 4);
  // Keep the point under the cursor fixed while zooming.
  const k = coverScale() / before;
  pan.x = (pan.x + e.clientX) * k - e.clientX;
  pan.y = (pan.y + e.clientY) * k - e.clientY;
  applyTransform();
}, { passive: false });

addEventListener('resize', applyTransform);

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

$('startBtn').addEventListener('click', async () => {
  const btn = $('startBtn');
  btn.disabled = true;
  try {
    $('status').textContent = 'loading…';
    scene = await (await fetch(SCENE_URL)).json();
    pano = scene.panorama;
    if (!pano) throw new Error('scene.json has no "panorama" block — run ingest_images.py');
    if (!scene.layers?.length) throw new Error('scene.json has no layers — run prepare_audio.py');
    pano.vfov_deg = pano.vfov_deg ?? pano.hfov_deg / 2;

    await loadImage(BASE_URL + pano.file);
    await engine.loadScene(scene, {
      baseUrl: BASE_URL,
      onProgress: (d, t) => { $('status').textContent = `loading sounds… ${d}/${t}`; },
    });
    await engine.start();

    // Open looking at the middle of the image, not at its left edge.
    const m = maxPan();
    pan = { x: m.x / 2, y: m.y / 2 };
    applyTransform();

    $('poiName').textContent = scene.name ?? scene.id;
    $('scenic').value = scene.scene?.scenicness ?? 8;
    $('scenicVal').textContent = (+$('scenic').value).toFixed(1);
    $('pressure').value = scene.scene?.pressure ?? 0;
    $('pressureVal').textContent = (+$('pressure').value).toFixed(1);
    buildRows();
    for (const el of ['title', 'meters', 'compass', 'controls']) $(el).hidden = false;
    $('gate').style.display = 'none';
    frame();
  } catch (err) {
    btn.disabled = false;
    $('status').textContent = 'headphones required';
    fail(`${err.message}<br>Serve the repo root over http (<code>npm run dev</code>) — ` +
         `ES modules and fetch don't work from file://`);
  }
});

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
    const row = document.createElement('div');
    row.className = 'row ' + l.type;
    row.innerHTML = `<span class="lbl">${l.id}</span><span class="bar"><i></i></span><span class="db">—</span>`;
    host.appendChild(row);
    rowEls.set(l.id, { row, fill: row.querySelector('i'), db: row.querySelector('.db') });
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
    const fresh = engine.ctx && engine.ctx.currentTime - l.lastEventAt < 0.6;
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
  for (const l of debug) {
    const el = rowEls.get(l.id);
    if (!el) continue;
    el.fill.style.width = (l.visibility * 100).toFixed(0) + '%';
    el.db.textContent = l.type === 'event'
      ? (l.visibility * 100).toFixed(0) + '%'
      : l.gainDb.toFixed(0);
    el.row.classList.toggle('fired', engine.ctx.currentTime - l.lastEventAt < 0.5);
  }
  drawCompass(debug, view);
  $('bearing').textContent = Math.round(view.yaw) + '°';
  $('fovOut').textContent = Math.round(view.fovH) + '°';
}
