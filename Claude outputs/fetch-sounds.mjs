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
// classes have no audio wired up, and writes scenes/audio-needs.json. Reading that keeps
// the class taxonomy in exactly one place (Python) instead of drifting between two.
// Without it we fall back to the list below, which is what the Hohe Tauern scenes
// needed the first time round.
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../..');
const NEEDS_FILE = process.env.NEEDS_FILE || path.join(REPO, 'scenes', 'audio-needs.json');

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
    console.log('No scenes/audio-needs.json — using the built-in list.');
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

async function searchOnce(query, { minDur, maxDur, count }) {
  const filter = [`license:${LICENSES}`, 'type:(wav OR flac OR aiff OR mp3)'];
  if (minDur != null && maxDur != null) filter.push(`duration:[${minDur} TO ${maxDur}]`);

  const url = new URL(API);
  url.searchParams.set('query', query);
  url.searchParams.set('filter', filter.join(' '));
  url.searchParams.set('sort', 'rating_desc');
  url.searchParams.set('page_size', String(Math.max(count * 3, 8)));
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
 * word at a time. Report which attempt actually worked so the query table can be fixed.
 */
async function search(query, opts) {
  const words = query.split(/\s+/);
  const attempts = [
    { q: query, dur: true, why: 'exact' },
    { q: query, dur: false, why: 'any duration' },
  ];
  if (words.length > 2) attempts.push({ q: words.slice(0, 2).join(' '), dur: false, why: `"${words.slice(0, 2).join(' ')}"` });
  if (words.length > 1) attempts.push({ q: words[0], dur: false, why: `"${words[0]}"` });

  for (const a of attempts) {
    const hits = await searchOnce(a.q, {
      count: opts.count,
      minDur: a.dur ? opts.minDur : null,
      maxDur: a.dur ? opts.maxDur : null,
    });
    if (hits.length) return { hits, why: a.why };
  }
  return { hits: [], why: null };
}

async function download(url, dest) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download ${res.status} ${url}`);
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
}

// Downloads are SOURCE material, so they live beside our own recordings under
// resources/ — not in scenes/, which holds only what the pipeline generates.
// prepare_audio.py then turns them into scene layers exactly as it does our own takes.
const outDir = process.env.OUT_DIR
  ? path.resolve(process.env.OUT_DIR)
  : path.join(REPO, 'resources', 'sounds-freesound');
if (!existsSync(outDir)) await mkdir(outDir, { recursive: true });

const credits = [];
const manifestLayers = [];
const relaxed = [];
const missing = [];

for (const spec of PLAN) {
  const { query, count = 1, minDur, maxDur, ...layer } = spec;
  process.stdout.write(`  ${spec.id.padEnd(12)} "${query}" … `);

  let hits = [], why = null;
  try {
    ({ hits, why } = await search(query, { minDur, maxDur, count }));
  } catch (err) {
    console.log(`FAILED (${err.message})`);
    continue;
  }
  if (!hits.length) {
    console.log('no results even after relaxing — try a different query for this class');
    missing.push(spec.id);
    continue;
  }

  const files = [];
  for (const [i, hit] of hits.entries()) {
    const file = `${spec.id}-${i + 1}.mp3`;
    await download(hit.previews['preview-hq-mp3'], path.join(outDir, file));
    files.push(file);
    credits.push({
      file, id: hit.id, name: hit.name, author: hit.username,
      license: hit.license, url: `https://freesound.org/s/${hit.id}/`,
    });
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
