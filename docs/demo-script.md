# The two-minute jury run

120 seconds, three beats, one ask. The whole thing is judged by ear, so **the room needs
to hear it** — a speaker or a headphone splitter beats a laptop speaker, and a headphone
pair passed to one judge beats nothing at all.

Keep the live mix panel visible for the entire demo. The jury believing that the coupling
is real depends on seeing the numbers move with the drag.

---

## Before you start

- [ ] `npm run dev` already running, page already loaded, gate showing. Never load live.
- [ ] Volume set and tested in the actual room.
- [ ] Scenicness slider at the scene's authored value; pressure slider at **0**.
- [ ] Browser zoom at 100 %, other tabs closed, notifications off.
- [ ] `overlay.png` open in a second tab as the fallback if audio dies.

---

## 0:00 — The gate (10 s)

Press **Listen**. Then **say nothing for three seconds** and let the scene establish.

> "This is the Wasserfallboden in the Hohe Tauern. We recorded it yesterday."

The silence is doing work. Don't fill it.

## 0:10 — Look-to-listen (35 s)

Drag slowly towards the water. Point at the meter panel as the bar fills.

> "The mix follows where you look. Nothing is being triggered — it's continuous, and it's
> our own recordings, placed on a sphere by direction and distance."

Keep dragging past it, slowly.

> "And it doesn't cut out behind you. You still hear the waterfall, the way you would
> standing there. A hard mute would feel like a bug."

Say the engineering line once, because it is the credible part:

> "No AI runs while you're listening. The analysis happens once, offline. In the browser
> it's a few hundred arithmetic operations — it works on an old phone, in a valley, with
> no signal."

## 0:45 — Season and mood (20 s)

Pull **scenicness** down.

> "Same recordings, different conditions. This is the location in bad weather — colder,
> windier, more space, fewer birds. One control vector, so a tourism board gets summer and
> November from one set of files."

Bring it back up.

## 1:05 — The care moment (25 s)

This is the beat that answers the third challenge question. Do not skip it.

Raise **human pressure** slowly from 0.

> "Now add people."

Let the footsteps and voices come up on their own for two seconds, then name what is
happening:

> "Those were always in the scene, held forty decibels down. And listen to the birds —
> they're dropping out. That's not a metaphor; that's what happens when a group arrives."

Bring it back to zero.

> "We can make you want to go. We can also let you hear what happens when everyone does —
> and then tell you to come in September, or take the bus."

## 1:30 — The business (20 s)

> "This is a white-label module for tourism boards. They already have the photos and the
> GPX. The layer nobody sells them is what the place sounds like. One trail, five points,
> licensed per season."

## 1:50 — The ask (10 s)

Name the one thing you want. Pick one and be specific:

- a pilot trail with one regional board, or
- access to a park's existing 360 imagery, or
- a recording week in the park with proper ambisonic kit.

---

## If something breaks

| Failure | What to do |
| --- | --- |
| No sound | You are probably not on `http://` — it must be served, not opened from a file. Say "one second", reload the tab you pre-warmed. |
| Audio starts then dies | Tab lost focus or the device muted. Click the page once and continue talking. |
| Image won't load | Talk over `overlay.png` in the second tab: "this is what the analysis sees". |
| Total failure | Describe the pressure mechanic in words and show `overlay.png`. The idea survives without the demo; the demo does not survive without the idea. |

## What not to do

- Don't explain the architecture. Nobody scores you on the frustum test.
- Don't apologise for the material. "We recorded it ourselves yesterday" is the strongest
  sentence you have in front of this jury — the brief handed out the recorders.
- Don't demo more than one location. Depth beats breadth at 120 seconds.
- Don't let scenicness make the mountain sound *ugly*. Austere, exposed, vast — the sound
  of a place that doesn't care whether you're there. Ugly is for the chairlift.
