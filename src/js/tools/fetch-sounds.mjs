#!/usr/bin/env node
/**
 * fetch-sounds.mjs — fill the GAPS in a scene from Freesound.
 *
 * Our own recordings come first: they are the reason this project is credible, and
 * prepare_audio.py turns them into the scene's beds and regions. This script is for the
 * sounds we did not manage to capture — a raven, a cowbell, a marmot — so the scene has
 * life in directions the recorder never pointed.
 *
 *   1. Get an API key (instant, no approval wait): https://freesound.org/apiv2/apply/
 *   2. FREESOUND_TOKEN=xxxx node fetch-sounds.mjs
 *
 * Downloads preview MP3s (no OAuth needed — only search requires the token),
 * writes ./audio/<poi>/*.mp3, a ready-to-load scene manifest, and CREDITS.md.
 *
 * Licence handling: by default this only pulls Creative Commons 0 sounds, so
 * there is nothing to clear before the demo. Pass --allow-by to also accept
 * Attribution sounds — those are fine too, but you must then ship CREDITS.md
 * somewhere visible in the app.
 */

import { writeFile, mkdir } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

const TOKEN = process.env.FREESOUND_TOKEN;
if (!TOKEN) {
  console.error('Set FREESOUND_TOKEN first. Get one instantly at https://freesound.org/apiv2/apply/');
  process.exit(1);
}

// CC0 alone is a small corner of Freesound, and combined with a duration filter it
// returns nothing for most specific queries. Attribution is equally safe commercially —
// it just has to be credited, and CREDITS.md is written automatically — so both are the
// default. --cc0-only restores the strict behaviour.
const CC0_ONLY = process.argv.includes('--cc0-only');
const LICENSES = CC0_ONLY
  ? '"Creative Commons 0"'
  : '("Creative Commons 0" OR "Attribution")';

/**
 * The scene spec. Each entry is both a Freesound query AND the layer geometry.
 * az = degrees clockwise from north, el = degrees up, distance = metres.
 * Tune az/el/distance to match whatever panorama you shoot — that is the whole
 * authoring job for a hackathon PoI, and it takes about ten minutes.
 */
// What to fetch comes from the scenes themselves when it can.
//
// `check_coverage.py --json` walks every scene's class report, works out which sounding
// classes have no audio wired up, and writes data/audio-needs.json. Reading that keeps
// the class taxonomy in exactly one place (Python) instead of drifting between two.
// Without it we fall back to the list below, which is what the Hohe Tauern scenes
// needed the first time round.
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../..');
// The generated data lives in 360viewer_app next door — see src/python/paths.py,
// which is what writes audio-needs.json in the first place. Same env override.
// (image_analysis and 360viewer_app are siblings under one project root now — this
// used to say `path.resolve(REPO, '..', 'SonicScapes', '360viewer_app', 'data')` back
// when the two were separate sibling repos named SoundScapes/SonicScapes. After they
// were folded into one root that doubled the "SonicScapes" segment and pointed
// nowhere; fixed to just go one level up, matching paths.py.)
const DATA_ROOT = process.env.SOUNDSCAPES_DATA ||
  path.resolve(REPO, '..', '360viewer_app', 'data');
const NEEDS_FILE = process.env.NEEDS_FILE || path.join(DATA_ROOT, 'audio-needs.json');

const FALLBACK = [
  { id: 'rock',    type: 'region', query: 'wind mountain ridge rock',      minDur: 15, maxDur: 90 },
  { id: 'snow',    type: 'region', query: 'wind over snow field',          minDur: 15, maxDur: 90 },
  { id: 'scree',   type: 'region', query: 'wind gravel scree slope',       minDur: 15, maxDur: 90 },
  { id: 'glacier', type: 'region', query: 'glacier ice creaking',          minDur: 10, maxDur: 90 },
  { id: 'pasture', type: 'region', query: 'alpine meadow insects summer',  minDur: 15, maxDur: 90 },
  { id: 'built',   type: 'region', query: 'wooden hut creak wind',         minDur: 10, maxDur: 60 },
  { id: 'cattle',  type: 'event',  query: 'cow bell alps',                 count: 3, maxDur: 8 },
  { id: 'animal',  type: 'event',  query: 'alpine chough bird call',       count: 3, maxDur: 6 },
  { id: 'person',  type: 'event',  query: 'distant hikers voices outdoor', count: 3, maxDur: 8 },
];

function loadPlan() {
  if (!existsSync(NEEDS_FILE)) {
    console.log('No data/audio-needs.json — using the built-in list.');
    console.log('For a list driven by what your scenes actually contain, run first:');
    console.log('  python src/python/audio_prep/check_coverage.py --json\n');
    return FALLBACK;
  }
  const { needs } = JSON.parse(readFileSync(NEEDS_FILE, 'utf8'));
  console.log(`${needs.length} class(es) missing audio, from ${path.relative(REPO, NEEDS_FILE)}:`);
  for (const n of needs) {
    console.log(`  ${String(n.max_share_percent).padStart(3)}%  ${n.class.padEnd(10)} `
      + `${n.type.padEnd(6)} "${n.query}"  (${n.scenes.length} scene(s))`);
  }
  console.log();
  return needs.map((n) => ({
    id: n.class,
    type: n.type,
    query: n.query,
    count: n.count ?? (n.type === 'event' ? 3 : 1),
    minDur: n.min_seconds ?? (n.type === 'event' ? 0 : 15),
    maxDur: n.max_seconds ?? (n.type === 'event' ? 8 : 90),
  }));
}

const PLAN = loadPlan();

const API = 'https://freesound.org/apiv2/search/text/';

// Alternate phrasings per class, tried after the primary query (from audio-needs.json
// / segment_panorama.py's QUERIES table) comes up empty even once duration and word
// count have been relaxed. A single hand-written phrase, however carefully worded, is a
// small net on a library the size of Freesound: "wind gusts exposed ridge" is exactly
// what the scene needs, but it is not necessarily how anyone tagged their recording.
// These give the search a different angle — a different noun, a more literal tag-style
// phrase, a near-synonym — before we give up on a class entirely. Matched by the
// class id (spec.id / n.class), so it lines up with QUERIES / SOUND_SPEC regardless of
// which query string ended up in audio-needs.json.
const ALTERNATES = {
  rock:      ['mountain ridge wind', 'wind blowing over rocks', 'exposed summit wind gusts',
              'howling wind mountain', 'strong wind ambience outdoor'],
  scree:     ['walking on scree slope', 'loose gravel slope wind', 'rockslide loose stones',
              'scree slope ambience', 'wind gravel mountain path'],
  snow:      ['wind blizzard snow', 'snowstorm wind gusts', 'arctic wind snow field',
              'wind over snowfield', 'winter mountain wind'],
  glacier:   ['glacier ice cracking', 'ice calving glacier', 'creaking ice sheet',
              'icefall rumble', 'ice cracking ambience'],
  forest:    ['pine forest wind ambience', 'alpine forest ambience', 'wind through trees',
              'coniferous forest ambience', 'forest ambience birds wind'],
  pasture:   ['summer meadow ambience', 'grasshoppers meadow field', 'alpine pasture ambience',
              'crickets summer field', 'insects meadow ambience'],
  cattle:    ['cowbell ambience', 'cow bells distant', 'alpine cows grazing bells',
              'sheep bell alps', 'cowbell single'],
  water:     ['alpine stream flowing', 'mountain brook water', 'small creek water flow',
              'babbling brook', 'stream water ambience'],
  built:     ['wooden cabin creak', 'wood cabin wind creak', 'mountain hut ambience',
              'timber structure creaking', 'wooden door creak wind'],
  animal:    ['alpine bird call', 'small bird single chirp', 'marmot whistle call',
              'chough bird call', 'bird chirping single'],
  waterfall: ['waterfall distant roar', 'small waterfall close', 'cascading water falls',
              'waterfall ambience loop'],
  person:    ['hikers talking distance', 'group hikers outdoor voices', 'trail chatter outdoor',
              'people talking outdoor distant'],
  cablecar:  ['cable car motor hum', 'chairlift mechanism hum', 'ski lift motor hum',
              'gondola lift mechanism'],
};

async function searchOnce(query, { minDur, maxDur, count, anyType }) {
  const filter = [`license:${LICENSES}`];
  if (!anyType) filter.push('type:(wav OR flac OR aiff OR mp3)');
  if (minDur != null && maxDur != null) filter.push(`duration:[${minDur} TO ${maxDur}]`);

  const url = new URL(API);
  url.searchParams.set('query', query);
  url.searchParams.set('filter', filter.join(' '));
  url.searchParams.set('sort', 'rating_desc');
  url.searchParams.set('page_size', String(Math.max(count * 4, 12)));
  url.searchParams.set('fields', 'id,name,username,license,duration,previews,avg_rating');
  url.searchParams.set('token', TOKEN);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Freesound ${res.status}: ${await res.text()}`);
  return ((await res.json()).results ?? []).slice(0, count);
}

/**
 * Try progressively looser searches rather than giving up.
 *
 * "wind gusts exposed ridge" with a 15-90 s duration filter matches nothing, which says
 * the query was too specific — not that Freesound has no mountain wind. So: relax the
 * duration filter first (it is the most arbitrary constraint), then shorten the query a
 * word at a time, then drop the original-file-type filter (previews are always mp3
 * regardless of source format, so this filter only ever narrows, never helps), and only
 * once the primary phrase is fully exhausted, move on to this class's ALTERNATES —
 * differently-worded searches for the same sound, each run through the same duration /
 * word-count / file-type ladder. Report which attempt actually worked so the query
 * table can be fixed.
 */
function ladder(query) {
  const words = query.split(/\s+/);
  const attempts = [
    { q: query, dur: true, anyType: false, why: 'exact' },
    { q: query, dur: false, anyType: false, why: 'any duration' },
    { q: query, dur: false, anyType: true, why: 'any duration, any file type' },
  ];
  if (words.length > 2) {
    const q = words.slice(0, 2).join(' ');
    attempts.push({ q, dur: false, anyType: true, why: `"${q}"` });
  }
  if (words.length > 1) {
    attempts.push({ q: words[0], dur: false, anyType: true, why: `"${words[0]}"` });
  }
  return attempts;
}

async function search(query, opts, classId) {
  const phrasings = [query, ...(ALTERNATES[classId] ?? [])];
  for (const [i, phrase] of phrasings.entries()) {
    for (const a of ladder(phrase)) {
      const hits = await searchOnce(a.q, {
        count: opts.count,
        minDur: a.dur ? opts.minDur : null,
        maxDur: a.dur ? opts.maxDur : null,
        anyType: a.anyType,
      });
      if (hits.length) {
        const why = i === 0 ? a.why : `alt "${phrase}" (${a.why})`;
        return { hits, why };
      }
    }
  }
  return { hits: [], why: null };
}

// The sandbox this was first written in has a well-behaved connection to Freesound;
// yours may not — a body timeout mid-download previously crashed the whole run with an
// uncaught exception (the download loop below wasn't wrapped in try/catch at all), which
// meant restarting from class #1 every time a single download hiccupped partway through
// a batch. retry() gives every network call a few attempts with backoff before it's
// allowed to fail, and every call site below now actually catches the final failure
// instead of letting it escape.
async function retry(fn, { attempts = 3, baseDelayMs = 1000 } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (i < attempts - 1) {
        await new Promise((r) => setTimeout(r, baseDelayMs * 2 ** i));
      }
    }
  }
  throw lastErr;
}

async function download(url, dest) {
  await retry(async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`download ${res.status} ${url}`);
    await writeFile(dest, Buffer.from(await res.arrayBuffer()));
  });
}

// Downloads are SOURCE material, so they live beside our own recordings under
// resources/ — not in data/scenes/, which holds only what the pipeline generates.
// prepare_audio.py then turns them into scene layers exactly as it does our own takes.
const outDir = process.env.OUT_DIR
  ? path.resolve(process.env.OUT_DIR)
  : path.join(REPO, 'resources', 'sounds-freesound');
if (!existsSync(outDir)) await mkdir(outDir, { recursive: true });

// Resume support: a class that already has at least one downloaded file is skipped
// unless --force is passed. Without this, re-running after a crash (or just to pick up
// a class that was fixed) re-searches and re-downloads everything from scratch again —
// wasteful, and it burns through the day's Freesound rate limit for no reason.
const FORCE = process.argv.includes('--force');
function alreadyFetched(id) {
  return existsSync(path.join(outDir, `${id}-1.mp3`));
}

const credits = [];
const manifestLayers = [];
const relaxed = [];
const missing = [];

for (const spec of PLAN) {
  const { query, count = 1, minDur, maxDur, ...layer } = spec;
  process.stdout.write(`  ${spec.id.padEnd(12)} "${query}" … `);

  if (!FORCE && alreadyFetched(spec.id)) {
    console.log('already have audio for this class (use --force to re-fetch)');
    continue;
  }

  let hits = [], why = null;
  try {
    ({ hits, why } = await retry(() => search(query, { minDur, maxDur, count }, spec.id),
                                  { attempts: 2 }));
  } catch (err) {
    console.log(`FAILED (${err.message})`);
    continue;
  }
  if (!hits.length) {
    console.log('no results even after relaxing and trying alternate phrasings');
    missing.push(spec.id);
    continue;
  }

  const files = [];
  for (const [i, hit] of hits.entries()) {
    const file = `${spec.id}-${i + 1}.mp3`;
    try {
      await download(hit.previews['preview-hq-mp3'], path.join(outDir, file));
    } catch (err) {
      console.log(`\n    [skipped one file: ${err.message}]`);
      continue;
    }
    files.push(file);
    credits.push({
      file, id: hit.id, name: hit.name, author: hit.username,
      license: hit.license, url: `https://freesound.org/s/${hit.id}/`,
    });
  }
  if (!files.length) {
    console.log('found hits but every download failed — try again');
    missing.push(spec.id);
    continue;
  }

  manifestLayers.push({ id: spec.id, type: spec.type, src: files });
  console.log(`${files.length} file(s) — ${hits[0].license}`
    + (why === 'exact' ? '' : `  [relaxed to ${why}]`));
  if (why !== 'exact') relaxed.push(`${spec.id}: ${why}`);
}

// Reference only — prepare_audio.py derives the real geometry from the filenames, and
// build_scene_layers.py replaces it with the geometry segmentation actually measured.
await writeFile(path.join(outDir, 'fetched.json'),
                JSON.stringify({ fetched: manifestLayers }, null, 2));

const creditsMd = [
  `# Audio credits — gap-fill downloads`,
  '',
  'Sourced from Freesound. Creative Commons 0 needs no attribution;',
  'Attribution-licensed sounds must be credited wherever the app is published.',
  '',
  '| File | Sound | Author | Licence |',
  '| --- | --- | --- | --- |',
  ...credits.map((c) => `| \`${c.file}\` | [${c.name}](${c.url}) | ${c.author} | ${c.license} |`),
  '',
].join('\n');
await writeFile(path.join(outDir, 'CREDITS.md'), creditsMd);

console.log(`\nDone. ${credits.length} files in ${outDir}`);
if (relaxed.length) {
  console.log(`\n${relaxed.length} query(ies) only matched after relaxing — worth editing`);
  console.log('QUERIES in src/python/image_analysis/segment_panorama.py:');
  for (const r of relaxed) console.log(`  ${r}`);
}
if (missing.length) {
  console.log(`\nStill nothing for: ${missing.join(', ')}`);
  console.log('Edit their QUERIES entries, or record them yourselves.');
}
console.log('\nNow turn them into scene layers, same as our own recordings:');
console.log('  python src/python/audio_prep/prepare_audio.py \\');
console.log('      --in resources/sounds-freesound --scene hohe-tauern\n');
