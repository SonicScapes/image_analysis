/**
 * SoundscapeEngine — viewport-coupled spatial audio mixer.
 *
 * Plays a set of pre-recorded loops and one-shots positioned on a sphere around
 * the listener, and continuously rebalances them according to where the camera
 * is looking and how far away each source is.
 *
 * Design rules (these are deliberate, don't "fix" them):
 *  - Looking at a source EMPHASISES it (default +9 dB). It never mutes what is
 *    behind you — a hard gate sounds like a bug, not like immersion.
 *  - All gain changes are smoothed: fast to rise, slow to fall, so panning the
 *    view swells the mix instead of switching it.
 *  - Distance sets both level and a low-pass. Far things are quiet AND dull.
 *
 * No dependencies. Framework agnostic. ~1 kB gzipped of actual logic.
 *
 * Usage:
 *   const engine = new SoundscapeEngine();
 *   await engine.loadScene(manifest, { baseUrl: '/audio/kaprun/' });
 *   await engine.start();                       // must be in a user gesture
 *   engine.setView({ yaw: 120, pitch: -5, fovH: 80, fovV: 55 });
 *   engine.getDebug();                          // per-layer vis + gain, for meters
 */

const DEG = Math.PI / 180;

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const dbToGain = (db) => Math.pow(10, db / 20);
const lerp = (a, b, t) => a + (b - a) * t;

/**
 * Resolve a layer's `src` against the scene's folder. Audio lives in one shared pool
 * (scenes/audio/) rather than being copied into every scene, so `src` is usually something
 * like "../../audio/water.mp3" — which string concatenation would mangle and URL
 * resolution handles correctly. Absolute paths and full URLs pass through untouched.
 */
function resolveSrc(src, baseUrl) {
  try {
    const base = new URL(baseUrl || './', window.location.href);
    return new URL(src, base).href;
  } catch {
    return (baseUrl || '') + src;      // non-browser contexts (tests)
  }
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0 || 1e-9), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Azimuth (degrees clockwise from north) + elevation (degrees up) to a Web Audio
 * vector. Web Audio is right-handed: +x right, +y up, -z forward. We define
 * north as forward, so az=0 -> (0,0,-1).
 */
function sphericalToVec(azDeg, elDeg) {
  const az = azDeg * DEG;
  const el = elDeg * DEG;
  const ce = Math.cos(el);
  return { x: ce * Math.sin(az), y: Math.sin(el), z: -ce * Math.cos(az) };
}

const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a, b) => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});

/** Defaults applied to every layer that doesn't override them. */
const LAYER_DEFAULTS = {
  type: 'region',   // 'bed' | 'region' | 'event' | 'score'
  az: 0,
  el: 0,
  spread: 40,       // angular half-extent, degrees
  distance: 50,     // metres
  gain: 0,          // base level, dB
  focus: 9,         // dB gained between "not in view" and "centred in view"
  loop: true,
  ratePerMin: 6,    // events only
  jitter: 0.7,      // events only: 0 = metronomic, 1 = fully random
  tags: [],         // 'wind' | 'human' | 'wildlife' — drive the macro controls
};

export class SoundscapeEngine {
  constructor(opts = {}) {
    /** @type {AudioContext|null} */
    this.ctx = null;
    this.scene = null;
    this.layers = [];
    this.buffers = new Map();

    this.view = { yaw: 0, pitch: 0, fovH: 80, fovV: 55 };
    this.mood = { scenicness: 6, eventfulness: 5 };
    // 0 = the place as we recorded it, 10 = the place under tourism pressure.
    this.pressure = 0;

    // Smoothing time constants for setTargetAtTime. Reaches ~95% at 3*tau.
    this.attackTau = opts.attackTau ?? 0.12;   // ~360 ms to settle upward
    this.releaseTau = opts.releaseTau ?? 0.6;  // ~1.8 s to settle downward

    // Reference distance in metres at which a source plays at its base gain.
    this.refDistance = opts.refDistance ?? 20;
    // Distance in metres over which the spectrum loses half its top octave.
    this.airHalfDistance = opts.airHalfDistance ?? 300;

    this.masterGainDb = opts.masterGainDb ?? 0;
    this.running = false;
    // Silent mode: geometry only. No AudioContext, no downloads, no graph — but
    // visibility, panning and the event scheduler all still run, so the meters and the
    // compass work. This is the mode for checking where you put things.
    this.silent = false;
    this._raf = null;
    this._lastUpdate = 0;
    this._updateHz = opts.updateHz ?? 20;
  }

  // ---------------------------------------------------------------- lifecycle

  /**
   * Load a PoI manifest. Decodes every referenced file up front so the scene
   * runs with no network afterwards.
   * @param {object} manifest
   * @param {{baseUrl?: string, onProgress?: (done:number,total:number)=>void}} [opts]
   */
  async loadScene(manifest, opts = {}) {
    const baseUrl = opts.baseUrl ?? '';
    this.silent = !!opts.silent;
    if (this.silent) return this._loadSceneSilent(manifest, baseUrl);
    this._ensureContext();

    const srcs = new Set();
    for (const l of manifest.layers) {
      for (const s of [].concat(l.src)) srcs.add(resolveSrc(s, baseUrl));
    }

    let done = 0;
    const total = srcs.size;
    await Promise.all(
      [...srcs].map(async (url) => {
        if (!this.buffers.has(url)) {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`SoundscapeEngine: cannot load ${url} (${res.status})`);
          const bytes = await res.arrayBuffer();
          this.buffers.set(url, await this.ctx.decodeAudioData(bytes));
        }
        opts.onProgress?.(++done, total);
      })
    );

    this.scene = { ...manifest, baseUrl };
    if (manifest.scene) {
      this.mood = {
        scenicness: manifest.scene.scenicness ?? this.mood.scenicness,
        eventfulness: manifest.scene.eventfulness ?? this.mood.eventfulness,
      };
      // Segmentation may have seen people in the frame. If so the scene starts with that
      // much human presence already audible, rather than at pristine.
      if (manifest.scene.pressure != null) {
        this.pressure = clamp(manifest.scene.pressure, 0, 10) / 10;
      }
    }
    this._buildGraph();
    return this;
  }

  /** Geometry-only load: no audio fetched, no graph built. */
  _loadSceneSilent(manifest, baseUrl) {
    this.scene = { ...manifest, baseUrl };
    if (manifest.scene) {
      this.mood = {
        scenicness: manifest.scene.scenicness ?? this.mood.scenicness,
        eventfulness: manifest.scene.eventfulness ?? this.mood.eventfulness,
      };
      if (manifest.scene.pressure != null) {
        this.pressure = clamp(manifest.scene.pressure, 0, 10) / 10;
      }
    }
    this.layers = manifest.layers.map((raw) => ({
      ...LAYER_DEFAULTS, ...raw, vis: 0, currentDb: -120, lastEventAt: 0,
    }));
    return this;
  }

  /** Seconds on whichever clock we have — there is no AudioContext in silent mode. */
  now() {
    return this._now();
  }

  _now() {
    return this.ctx ? this.ctx.currentTime : performance.now() / 1000;
  }

  /** Resume the AudioContext and start playback. Call from a click/tap handler. */
  async start() {
    if (this.silent) {
      if (this.running) return this;
      this.running = true;
      for (const layer of this.layers) {
        if (layer.type === 'event') this._scheduleEvent(layer);
      }
      this._tick();
      return this;
    }
    this._ensureContext();
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    if (this.running) return this;
    this.running = true;

    const t = this.ctx.currentTime;
    for (const layer of this.layers) {
      if (layer.type === 'event') {
        this._scheduleEvent(layer);
      } else {
        layer.source = this.ctx.createBufferSource();
        layer.source.buffer = layer.buffer;
        layer.source.loop = true;
        layer.source.connect(layer.input);
        // Random start offset so two visits never phase-align.
        layer.source.start(t, Math.random() * layer.buffer.duration);
      }
    }
    this._tick();
    this.master.gain.setTargetAtTime(dbToGain(this.masterGainDb), t, 0.4);
    return this;
  }

  /** Fade out and stop everything. */
  async stop(fadeSeconds = 0.6) {
    if (!this.running) return;
    this.running = false;
    if (this.silent) {
      for (const l of this.layers) if (l.timer) clearTimeout(l.timer);
      cancelAnimationFrame(this._raf);
      return;
    }
    const t = this.ctx.currentTime;
    this.master.gain.setTargetAtTime(0.0001, t, fadeSeconds / 3);
    await new Promise((r) => setTimeout(r, fadeSeconds * 1000));
    for (const layer of this.layers) {
      layer.source?.stop();
      layer.source = null;
      if (layer.timer) clearTimeout(layer.timer);
    }
    cancelAnimationFrame(this._raf);
  }

  /** Master level in dB, for ducking under a voiceover or a UI moment. */
  setMasterGain(db, seconds = 0.3) {
    this.masterGainDb = db;
    if (this.silent || !this.ctx) return;
    this.master?.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, seconds / 3);
  }

  // -------------------------------------------------------------------- input

  /**
   * Feed the camera in. Call this every frame; it's cheap and internally throttled.
   * @param {{yaw:number, pitch:number, fovH?:number, fovV?:number}} view degrees
   */
  setView(view) {
    this.view.yaw = view.yaw;
    this.view.pitch = view.pitch;
    if (view.fovH != null) this.view.fovH = view.fovH;
    if (view.fovV != null) this.view.fovV = view.fovV;
  }

  /**
   * The two perception axes (ISO 12913 circumplex). Drives the macro chain:
   * spectral tilt, reverb amount, event density, score level.
   * @param {{scenicness?:number, eventfulness?:number}} mood each 0..10
   */
  setMood(mood) {
    if (mood.scenicness != null) this.mood.scenicness = clamp(mood.scenicness, 0, 10);
    if (mood.eventfulness != null) this.mood.eventfulness = clamp(mood.eventfulness, 0, 10);
    this._applyMood();
  }

  /**
   * Human pressure, 0-10. Raises anything tagged 'human' (lift, road, crowd, footsteps)
   * and makes wildlife go quiet — which is what actually happens when people arrive, and
   * is the honest way to let someone HEAR why a place is worth protecting.
   */
  setPressure(p) {
    this.pressure = clamp(p, 0, 10) / 10;
    this._applyMood();
  }

  // ---------------------------------------------------------------- traversal

  /**
   * Move to the next PoI: equal-power crossfade of material, linear morph of mood.
   * This is the "traverse" moment — interpolate PARAMETERS, crossfade MATERIAL.
   * Never try to morph the audio content itself; it sounds like neither place.
   */
  async crossfadeTo(manifest, opts = {}) {
    const seconds = opts.seconds ?? 6;
    const baseUrl = opts.baseUrl ?? '';
    const outgoing = { layers: this.layers, master: this.master };
    const fromMood = { ...this.mood };

    // Build the incoming scene on its own master, silent.
    const incomingMaster = this.ctx.createGain();
    incomingMaster.gain.value = 0.0001;
    incomingMaster.connect(this.tilt);
    this.layers = [];
    this.master = incomingMaster;
    await this.loadScene(manifest, { baseUrl });
    await this.start();

    const toMood = { ...this.mood };
    const t0 = this.ctx.currentTime;
    outgoing.master.gain.setTargetAtTime(0.0001, t0, seconds / 4);
    incomingMaster.gain.setTargetAtTime(dbToGain(this.masterGainDb), t0, seconds / 4);

    // Morph the mood across the traverse so the transition feels continuous.
    const started = performance.now();
    const morph = () => {
      const k = clamp((performance.now() - started) / (seconds * 1000), 0, 1);
      this.setMood({
        scenicness: lerp(fromMood.scenicness, toMood.scenicness, k),
        eventfulness: lerp(fromMood.eventfulness, toMood.eventfulness, k),
      });
      if (k < 1) requestAnimationFrame(morph);
    };
    morph();

    setTimeout(() => {
      for (const l of outgoing.layers) {
        l.source?.stop();
        if (l.timer) clearTimeout(l.timer);
      }
      outgoing.master.disconnect();
    }, seconds * 1000 + 500);
    return this;
  }

  // ------------------------------------------------------------------- debug

  /** Per-layer state, for meters / an overlay. Cheap enough to call every frame. */
  getDebug() {
    const now = this._now();
    return this.layers.map((l) => {
      // An event only contributes to the mix while it is actually sounding.
      const firing = l.type === 'event' && now - (l.lastEventAt ?? 0) < 1.5;
      const db = l.type === 'event'
        ? (firing ? this._layerDb(l, l.vis ?? 0) : -120)
        : (l.currentDb ?? -120);
      return {
        id: l.id,
        type: l.type,
        az: l.az,
        el: l.el,
        distance: l.distance,
        visibility: l.vis ?? 0,
        gainDb: l.currentDb ?? -120,
        // Power, for working out each layer's share of what you are hearing.
        power: db > -60 ? Math.pow(10, db / 10) : 0,
        firing,
        files: (l.srcList ?? [].concat(l.src ?? [])).map(
          (u) => String(u).split('/').pop().split('?')[0]),
        lastEventAt: l.lastEventAt ?? 0,
      };
    });
  }

  // ----------------------------------------------------------------- internal

  _ensureContext() {
    if (this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC({ latencyHint: 'interactive' });

    // Master chain: [layers] -> master -> tilt -> (dry + reverb) -> limiter -> out
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.0001;

    this.tilt = this.ctx.createBiquadFilter();
    this.tilt.type = 'highshelf';
    this.tilt.frequency.value = 2000;
    this.tilt.gain.value = 0;

    this.dry = this.ctx.createGain();
    this.wet = this.ctx.createGain();
    this.wet.gain.value = 0.15;
    this.reverb = this.ctx.createConvolver();
    this.reverb.buffer = this._makeImpulse(2.2, 2.6);

    this.limiter = this.ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -6;
    this.limiter.ratio.value = 12;
    this.limiter.attack.value = 0.004;
    this.limiter.release.value = 0.25;

    this.master.connect(this.tilt);
    this.tilt.connect(this.dry);
    this.tilt.connect(this.reverb);
    this.reverb.connect(this.wet);
    this.dry.connect(this.limiter);
    this.wet.connect(this.limiter);
    this.limiter.connect(this.ctx.destination);

    const L = this.ctx.listener;
    if (L.positionX) {
      L.positionX.value = 0; L.positionY.value = 0; L.positionZ.value = 0;
    } else {
      L.setPosition(0, 0, 0);
    }
  }

  /** Cheap synthetic impulse response — swap for a real IR when you have one. */
  _makeImpulse(seconds, decay) {
    const rate = this.ctx.sampleRate;
    const len = Math.floor(rate * seconds);
    const buf = this.ctx.createBuffer(2, len, rate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) {
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
      }
    }
    return buf;
  }

  _buildGraph() {
    this.layers = this.scene.layers.map((raw) => {
      const l = { ...LAYER_DEFAULTS, ...raw };
      l.srcList = [].concat(l.src).map((s) => resolveSrc(s, this.scene.baseUrl));
      l.buffer = this.buffers.get(l.srcList[0]);

      // Per-layer chain: input -> airLP -> [panner | direct] -> gain -> master
      l.input = this.ctx.createGain();
      l.air = this.ctx.createBiquadFilter();
      l.air.type = 'lowpass';
      l.air.frequency.value = 20000;
      l.air.Q.value = 0.4;
      l.gainNode = this.ctx.createGain();
      l.gainNode.gain.value = 0.0001;

      l.input.connect(l.air);

      const isDiffuse = l.type === 'bed' || l.type === 'score';
      if (isDiffuse) {
        l.air.connect(l.gainNode);
      } else {
        // Wide regions are partly enveloping: blend spatialised and direct paths.
        const spatialAmount = clamp(1 - (l.spread - 30) / 90, 0.25, 1);
        l.panner = this.ctx.createPanner();
        l.panner.panningModel = 'HRTF';
        l.panner.distanceModel = 'inverse';
        l.panner.refDistance = 1;
        l.panner.rolloffFactor = 0; // we do distance ourselves, in dB
        const v = sphericalToVec(l.az, l.el);
        if (l.panner.positionX) {
          l.panner.positionX.value = v.x;
          l.panner.positionY.value = v.y;
          l.panner.positionZ.value = v.z;
        } else {
          l.panner.setPosition(v.x, v.y, v.z);
        }
        l.spatialGain = this.ctx.createGain();
        l.spatialGain.gain.value = spatialAmount;
        l.directGain = this.ctx.createGain();
        l.directGain.gain.value = 1 - spatialAmount;

        l.air.connect(l.spatialGain);
        l.spatialGain.connect(l.panner);
        l.panner.connect(l.gainNode);
        l.air.connect(l.directGain);
        l.directGain.connect(l.gainNode);
      }

      l.gainNode.connect(this.master);
      l.vis = 0;
      l.currentDb = -120;
      return l;
    });
    this._applyMood();
  }

  /** Visible fraction of this layer, 0..1, as an elliptical frustum test. */
  _visibility(layer) {
    const f = sphericalToVec(this.view.yaw, this.view.pitch);
    const u = sphericalToVec(this.view.yaw, this.view.pitch + 90);
    const r = cross(f, u);
    const v = sphericalToVec(layer.az, layer.el);

    const xl = dot(v, r);
    const yl = dot(v, u);
    const zl = dot(v, f);

    const uAng = Math.atan2(xl, zl) / DEG;                       // horizontal offset
    const vAng = Math.atan2(yl, Math.hypot(xl, zl)) / DEG;       // vertical offset

    const halfH = this.view.fovH / 2;
    const halfV = this.view.fovV / 2;
    // Normalised elliptical distance: 1.0 sits exactly on the frustum edge.
    const d = Math.hypot(uAng / halfH, vAng / halfV);
    const spreadN = layer.spread / halfH;

    return smoothstep(1 + spreadN, Math.max(1 - spreadN, 0), d);
  }

  _tick = () => {
    if (!this.running) return;
    this._raf = requestAnimationFrame(this._tick);

    const now = performance.now();
    if (now - this._lastUpdate < 1000 / this._updateHz) return;
    this._lastUpdate = now;

    const t = this._now();

    if (this.silent) {
      for (const layer of this.layers) {
        layer.vis = layer.type === 'bed' || layer.type === 'score'
          ? 1 : this._visibility(layer);
        layer.currentDb = this._layerDb(layer, layer.vis);
      }
      return;
    }

    // Listener follows the camera; sources stay put in world space.
    const f = sphericalToVec(this.view.yaw, this.view.pitch);
    const u = sphericalToVec(this.view.yaw, this.view.pitch + 90);
    const L = this.ctx.listener;
    if (L.forwardX) {
      L.forwardX.value = f.x; L.forwardY.value = f.y; L.forwardZ.value = f.z;
      L.upX.value = u.x;      L.upY.value = u.y;      L.upZ.value = u.z;
    } else {
      L.setOrientation(f.x, f.y, f.z, u.x, u.y, u.z);
    }

    for (const layer of this.layers) {
      const vis = layer.type === 'bed' || layer.type === 'score' ? 1 : this._visibility(layer);
      layer.vis = vis;

      if (layer.type === 'event') continue; // events set their gain at trigger time

      const db = this._layerDb(layer, vis);
      const rising = db > layer.currentDb;
      layer.currentDb = db;
      layer.gainNode.gain.setTargetAtTime(
        dbToGain(db), t, rising ? this.attackTau : this.releaseTau
      );
      layer.air.frequency.setTargetAtTime(this._airCutoff(layer.distance), t, 0.2);
    }
  };

  _layerDb(layer, vis) {
    const distDb = -20 * Math.log10(Math.max(layer.distance, 1) / this.refDistance);
    const focusDb = -layer.focus * (1 - vis); // 0 when centred, -focus when out of view
    const scoreDuck = layer.type === 'score' ? -this._eventActivity() * 4 : 0;
    // Human layers are authored at their "busy day" level and held down until pressure
    // rises, so the slider reveals them rather than merely turning them up.
    const pressureDb = layer.tags?.includes('human') ? lerp(-40, 0, this.pressure) : 0;
    return clamp(layer.gain + distDb + focusDb + scoreDuck + pressureDb, -60, 12);
  }

  _airCutoff(distanceM) {
    // Half the top octave lost per airHalfDistance metres. Far = quiet AND dull.
    return clamp(20000 * Math.pow(0.5, distanceM / this.airHalfDistance), 700, 20000);
  }

  _eventActivity() {
    const now = this._now();
    let a = 0;
    for (const l of this.layers) {
      if (l.type === 'event' && l.lastEventAt && now - l.lastEventAt < 1.5) a += 0.5;
    }
    return clamp(a, 0, 1);
  }

  _scheduleEvent(layer) {
    if (!this.running) return;
    const eventfulness = this.mood.eventfulness / 5; // 1.0 at the neutral midpoint
    // Wildlife falls silent as people arrive; human events get more frequent.
    const wild = layer.tags?.includes('wildlife') ? 1 - 0.85 * this.pressure : 1;
    const human = layer.tags?.includes('human') ? 0.15 + 1.85 * this.pressure : 1;
    // Visible sources are more talkative, but nothing ever goes fully silent.
    const rate = Math.max(
      layer.ratePerMin * (0.4 + 0.6 * (layer.vis ?? 0)) * eventfulness * wild * human, 0.05);
    const mean = 60 / rate;
    const delay = mean * (1 - layer.jitter + layer.jitter * (0.3 + Math.random() * 1.7));

    layer.timer = setTimeout(() => {
      this._fireEvent(layer);
      this._scheduleEvent(layer);
    }, delay * 1000);
  }

  _fireEvent(layer) {
    if (!this.running) return;
    if (this.silent) {
      // No sound, but record the trigger so the meters and compass still blink.
      layer.lastEventAt = this._now();
      layer.lastEventAz = layer.az + (Math.random() * 2 - 1) * layer.spread;
      return;
    }
    const url = layer.srcList[(Math.random() * layer.srcList.length) | 0];
    const buffer = this.buffers.get(url);
    if (!buffer) return;

    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.playbackRate.value = 0.94 + Math.random() * 0.12; // slight natural variation

    // Place it somewhere inside the layer's angular extent, not always dead centre.
    const az = layer.az + (Math.random() * 2 - 1) * layer.spread;
    const el = layer.el + (Math.random() * 2 - 1) * layer.spread * 0.35;
    const v = sphericalToVec(az, el);

    const panner = this.ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    panner.refDistance = 1;
    panner.rolloffFactor = 0;
    if (panner.positionX) {
      panner.positionX.value = v.x; panner.positionY.value = v.y; panner.positionZ.value = v.z;
    } else {
      panner.setPosition(v.x, v.y, v.z);
    }

    const air = this.ctx.createBiquadFilter();
    air.type = 'lowpass';
    air.frequency.value = this._airCutoff(layer.distance);

    const g = this.ctx.createGain();
    g.gain.value = dbToGain(this._layerDb(layer, layer.vis ?? 0));

    src.connect(air); air.connect(panner); panner.connect(g); g.connect(this.master);
    src.start(t);
    src.onended = () => { g.disconnect(); panner.disconnect(); air.disconnect(); };

    layer.lastEventAt = t;
    layer.lastEventAz = az;
  }

  _applyMood() {
    if (this.silent) {
      // Keep the numbers current so the meters reflect the sliders.
      const s2 = this.mood.scenicness / 10;
      for (const l of this.layers) {
        if (l.tags?.includes('wind')) l.gain = lerp(-3, -14, s2);
        if (l.type === 'score') l.gain = lerp(-30, -24, s2);
      }
      return;
    }
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const s = this.mood.scenicness / 10; // 0 austere .. 1 pleasant

    // Six macros, collapsed to what a hackathon demo can actually hear:
    this.tilt.gain.setTargetAtTime(lerp(-4, 2, s), t, 0.5);            // spectral tilt
    this.wet.gain.setTargetAtTime(lerp(0.28, 0.12, s), t, 0.5);        // reverb amount
    // Pressure also flattens the sense of space: a busy place sounds smaller.
    this.reverbTrim = lerp(1, 0.55, this.pressure);
    this.wet.gain.setTargetAtTime(
      lerp(0.28, 0.12, s) * this.reverbTrim, t, 0.5);
    for (const l of this.layers) {
      if (l.id === 'wind' || l.tags?.includes('wind')) {
        l.gain = lerp(-3, -14, s);                                      // wind presence
      }
      if (l.type === 'score') {
        l.gain = lerp(-30, -24, s);                                     // score level
      }
    }
  }
}

export default SoundscapeEngine;
