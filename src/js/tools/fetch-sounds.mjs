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
import { existsSync } from 'node:fs';
import path from 'node:path';

const TOKEN = process.env.FREESOUND_TOKEN;
if (!TOKEN) {
  console.error('Set FREESOUND_TOKEN first. Get one instantly at https://freesound.org/apiv2/apply/');
  process.exit(1);
}

const ALLOW_BY = process.argv.includes('--allow-by');
const LICENSES = ALLOW_BY
  ? '("Creative Commons 0" OR "Attribution")'
  : '"Creative Commons 0"';

/**
 * The scene spec. Each entry is both a Freesound query AND the layer geometry.
 * az = degrees clockwise from north, el = degrees up, distance = metres.
 * Tune az/el/distance to match whatever panorama you shoot — that is the whole
 * authoring job for a hackathon PoI, and it takes about ten minutes.
 */
const SCENE = {
  id: 'kaprun-wasserfallboden',
  name: 'Wasserfallboden',
  scene: { scenicness: 8.4, eventfulness: 6.8 },
  layers: [
    { id: 'bed', type: 'bed', query: 'mountain wind ambience', minDur: 20, maxDur: 120, gain: -12 },
    { id: 'wind', type: 'region', query: 'wind gusts exposed ridge', minDur: 15, maxDur: 90,
      az: 320, el: 15, spread: 90, distance: 30, gain: -10, tags: ['wind'] },
    { id: 'waterfall', type: 'region', query: 'waterfall close', minDur: 15, maxDur: 90,
      az: 285, el: -10, spread: 30, distance: 110, gain: 2, focus: 10 },
    { id: 'stream', type: 'region', query: 'mountain stream brook', minDur: 15, maxDur: 90,
      az: 240, el: -25, spread: 45, distance: 35, gain: -4 },
    { id: 'forest', type: 'region', query: 'forest wind trees leaves ambience', minDur: 15, maxDur: 90,
      az: 150, el: -5, spread: 55, distance: 180, gain: -6 },
    { id: 'lake', type: 'region', query: 'lake shore water lapping', minDur: 15, maxDur: 90,
      az: 20, el: -18, spread: 50, distance: 260, gain: -8 },
    { id: 'meadow', type: 'region', query: 'meadow grasshopper insects summer', minDur: 15, maxDur: 90,
      az: 95, el: -12, spread: 60, distance: 60, gain: -12 },

    // Events. Keep 2-4 variants each so repetition never becomes audible.
    { id: 'cowbell', type: 'event', query: 'cow bell alps', count: 3, maxDur: 8,
      az: 110, el: -10, spread: 35, distance: 300, gain: 4, ratePerMin: 7 },
    { id: 'raven', type: 'event', query: 'raven croak call', count: 3, maxDur: 6,
      az: 300, el: 30, spread: 60, distance: 200, gain: 2, ratePerMin: 3 },
    { id: 'chough', type: 'event', query: 'alpine chough bird call', count: 2, maxDur: 6,
      az: 340, el: 40, spread: 70, distance: 420, gain: 4, ratePerMin: 2.5 },
    { id: 'marmot', type: 'event', query: 'marmot whistle', count: 2, maxDur: 5,
      az: 200, el: -5, spread: 40, distance: 150, gain: 3, ratePerMin: 1.5 },
    { id: 'rockfall', type: 'event', query: 'small rockfall gravel scree', count: 3, maxDur: 8,
      az: 315, el: 5, spread: 45, distance: 380, gain: 2, ratePerMin: 1.2 },
    { id: 'smallbirds', type: 'event', query: 'small bird chirp single', count: 4, maxDur: 5,
      az: 150, el: 5, spread: 70, distance: 45, gain: -4, ratePerMin: 12 },
  ],
};

const API = 'https://freesound.org/apiv2/search/text/';

async function search(query, { minDur = 0, maxDur = 60, count = 1 }) {
  const filter = [
    `license:${LICENSES}`,
    `duration:[${minDur} TO ${maxDur}]`,
    'type:(wav OR flac OR aiff OR mp3)',
  ].join(' ');

  const url = new URL(API);
  url.searchParams.set('query', query);
  url.searchParams.set('filter', filter);
  url.searchParams.set('sort', 'rating_desc');
  url.searchParams.set('page_size', String(Math.max(count * 3, 8)));
  url.searchParams.set('fields', 'id,name,username,license,duration,previews,avg_rating');
  url.searchParams.set('token', TOKEN);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Freesound ${res.status}: ${await res.text()}`);
  const json = await res.json();
  return (json.results ?? []).slice(0, count);
}

async function download(url, dest) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download ${res.status} ${url}`);
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
}

const here = path.dirname(new URL(import.meta.url).pathname);
const outDir = process.env.OUT_DIR
  ? path.resolve(process.env.OUT_DIR)
  : path.resolve(here, '../../../data/scenes', SCENE.id);
if (!existsSync(outDir)) await mkdir(outDir, { recursive: true });

const credits = [];
const manifestLayers = [];

for (const spec of SCENE.layers) {
  const { query, count = 1, minDur, maxDur, ...layer } = spec;
  process.stdout.write(`  ${spec.id.padEnd(12)} "${query}" … `);

  let hits = [];
  try {
    hits = await search(query, { minDur, maxDur, count });
  } catch (err) {
    console.log(`FAILED (${err.message})`);
    continue;
  }
  if (!hits.length) {
    console.log('no results — widen the query or add --allow-by');
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

  manifestLayers.push({ ...layer, src: files.length === 1 ? files[0] : files });
  console.log(`${files.length} file(s) — ${hits[0].license}`);
}

const manifest = {
  id: SCENE.id,
  name: SCENE.name,
  north_offset_deg: 0,
  scene: SCENE.scene,
  layers: manifestLayers,
};

await writeFile(path.join(outDir, 'scene.json'), JSON.stringify(manifest, null, 2));

const creditsMd = [
  `# Audio credits — ${SCENE.name}`,
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
console.log(`Scene manifest: ${path.join(outDir, 'scene.json')}`);
console.log(`Merge these layers into scene.json, then open src/js/app/index.html\n`);
