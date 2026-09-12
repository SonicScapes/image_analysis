# How this answers the challenge

Challenge 1 — *Experience Sound & Nature*, cta (change tourism austria) / Austria Tourism.

The jury is the tourism body that wrote the brief, and they will score against **their own
three questions**, not against our architecture. Worth reading this before the pitch,
because question 3 is the one we nearly missed.

---

## Q1 — "How can sounds from nature be recorded, mixed or transformed into an experience people can interact with?"

This is our strong suit and it is a literal description of what we built.

- **Recorded**: our own takes from the site with the handheld recorders, in
  `resources/sounds/`. Not library material, not generated — the actual place.
- **Mixed**: `soundscape-engine.js` is a real-time spatial mixer. Every recording becomes
  a direction, a width and a distance on a sphere; the mix rebalances forty times a second
  against where the visitor is looking.
- **Interact with**: you drag, or you turn your phone, and the soundscape answers. The
  interaction is *looking*, which is the one thing a tourist already does.

Say the number on stage: **no AI runs at runtime**. The analysis happens once, offline; the
browser does a frustum test. That is why it works on a phone, in a valley, with no signal.

## Q2 — "How can sound help tell stories about landscape, seasons, weather, tourism and sustainability?"

The engine carries a **control vector** separate from the audio material, which is what
makes storytelling possible at all: the same recordings can express different conditions
because the parameters move, not the files.

- **Landscape** is told by direction and distance. Turn towards the cirque and you hear its
  reverberation; turn towards the scree and the top end drops away.
- **Seasons and weather** are the scenicness axis plus stem swaps: the same location in
  November is a thinner event set, a colder spectral tilt, a longer reverb, more wind.
- **Tourism** is the human-pressure control (below).

Grounding worth mentioning if a judge pushes: the two axes are **ISO 12913-3**'s circumplex
of soundscape perception — pleasant↔annoying and eventful↔uneventful. We did not invent a
feeling scale; we used the one the soundscape research field standardised.

## Q3 — "How can the result encourage people to treat nature with more care?"

**This was the gap.** A beautiful immersive soundscape makes people want to *visit*, which
is the opposite of care if it stops there. The brief asks for more, and the jury will
notice if we only answer the first two questions.

Our answer is the **human-pressure control**, and it works because it is honest rather than
preachy:

- Raise it and the sounds of people appear — footsteps, voices, infrastructure. They were
  always in the scene, authored at their busy-day level and held 40 dB down.
- At the same time **wildlife goes quiet**. Event rates for anything tagged `wildlife` drop
  by up to 85 %. That is not a metaphor; that is what actually happens when people arrive.
- The sense of space collapses too: reverb trims back, because a busy place sounds smaller.

So the visitor does not get told that nature is fragile. They *hear* what their own presence
costs, in the same medium they just fell in love with. Then the app can ask the useful
question — go in the shoulder season, go early, take the bus — and it lands, because the
argument was made in sound rather than in a banner.

One line for the pitch: **"We can make you want to go. We can also let you hear what
happens when everyone does."**

---

## What to be careful about on stage

**Don't make the mountain sound bad.** Our original idea was "unpleasant soundscape when
the image is less beautiful". Pointed at a glacier that is a product no tourism board will
license. Pointed at *human impact*, the same mechanism is exactly what the brief asked for.
Keep the low end of the scenicness axis as **austere, exposed, vast** — the sound of high
places that do not care whether you are there — and save genuine unpleasantness for the
chairlift and the crowded summit.

**The available-tools line said handheld recorders.** We used them. Say so. A team that
went out and recorded the park beats a team that prompted a model, in front of this jury.

**Sustainability has to be in the two minutes, not the appendix.** It is a third of the
brief. Budget 20 seconds of the 120 for the pressure moment.

## Scoring checklist

| Brief asks for | We have | Where |
| --- | --- | --- |
| Sounds from nature, recorded | Own takes from the site | `resources/sounds/` |
| Mixed / transformed | Real-time spatial mixer | `src/js/engine/` |
| Interactive experience | Look-to-listen, drag or gyro | `src/js/app/` |
| Landscape storytelling | Direction, distance, reverb per region | `scene.json` geometry |
| Seasons & weather | Scenicness axis + stem swaps | `engine.setMood()` |
| Tourism | Human-pressure control | `engine.setPressure()` |
| Sustainability / care | Wildlife falls silent as pressure rises | `engine.setPressure()` |
| Tourism context | White-label module for boards, per trail | `docs/architecture.md` |
