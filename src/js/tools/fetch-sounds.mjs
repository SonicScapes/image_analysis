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
  id: 'gap-fill',
  name: 'Gap fill',
  scene: { scenicness: 7.5, eventfulness: 5.0 },
  // Targeted at what segmentation ACTUALLY found in the Hohe Tauern scenes — rock,
  // snow, scree and glacier dominate, and we recorded none of them. Water and waterfall
  // are deliberately absent: we have our own, and ours are better.
  //
  // Percentages are the share of the view each class covered, from scene_classes.txt.
  // Run src/python/audio_prep/check_coverage.py to regenerate this list for new scenes.
  layers: [
    // rock — up to 40% of the view. The single biggest hole.
    { id: 'rock', type: 'region', query: 'wind mountain ridge rock', minDur: 15, maxDur: 90,
      az: 0, el: 5, spread: 70, distance: 400, gain: -14, focus: 6, tags: ['wind'] },
    // snow — 18-26% in every scene, and we have nothing at all for it.
    { id: 'snow', type: 'region', query: 'wind over snow field', minDur: 15, maxDur: 90,
      az: 60, el: 0, spread: 60, distance: 300, gain: -16, focus: 6 },
    { id: 'scree', type: 'region', query: 'wind gravel scree slope', minDur: 15, maxDur: 90,
      az: 300, el: -10, spread: 55, distance: 250, gain: -12, focus: 7 },
    { id: 'glacier', type: 'region', query: 'glacier ice creaking', minDur: 10, maxDur: 90,
      az: 20, el: 0, spread: 40, distance: 500, gain: -14, focus: 7 },
    { id: 'pasture', type: 'region', query: 'alpine meadow insects summer', minDur: 15, maxDur: 90,
      az: 150, el: -12, spread: 60, distance: 60, gain: -12, focus: 7 },
    { id: 'built', type: 'region', query: 'wooden hut creak wind', minDur: 10, maxDur: 60,
      az: 200, el: -5, spread: 30, distance: 150, gain: -18, focus: 8 },

    // Events. 2-4 variants each so repetition never becomes audible.
    { id: 'cattle', type: 'event', query: 'cow bell alps', count: 3, maxDur: 8,
      az: 150, el: -8, spread: 40, distance: 300, gain: 4, ratePerMin: 6, tags: ['wildlife'] },
    { id: 'animal', type: 'event', query: 'alpine chough bird call', count: 3, maxDur: 6,
      az: 330, el: 30, spread: 70, distance: 250, gain: 2, ratePerMin: 3, tags: ['wildlife'] },
    { id: 'marmot', type: 'event', query: 'marmot whistle', count: 2, maxDur: 5,
      az: 260, el: -5, spread: 40, distance: 150, gain: 3, ratePerMin: 1.5, tags: ['wildlife'] },
    { id: 'rockfall', type: 'event', query: 'small rockfall gravel', count: 3, maxDur: 8,
      az: 300, el: 0, spread: 45, distance: 380, gain: 2, ratePerMin: 1.2 },
    // Human presence, for the pressure control. Held 40 dB down until it is raised.
    { id: 'cablecar', type: 'event', query: 'ski lift cable car motor', count: 2, maxDur: 10,
      az: 60, el: 8, spread: 40, distance: 250, gain: -2, ratePerMin: 4, tags: ['human'] },
    { id: 'person', type: 'event', query: 'distant hikers voices outdoor', count: 3, maxDur: 8,
      az: 200, el: -5, spread: 50, distance: 60, gain: -4, ratePerMin: 5, tags: ['human'] },
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
