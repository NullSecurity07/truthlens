# TruthLens — design and trust UX

Three surfaces share one design system: the **dashboard** (`localhost:5173`), the
**extension popup**, and the **in-page overlay**. This document records what was chosen and
why, so the interface can be defended the same way the engineering can.

---

## 1. The register: instrument, not product page

A verification tool has an unusual constraint — **marketing polish reduces trust.** Gradients,
hero animations and confident superlatives are the visual grammar of persuasion, and a tool
whose entire job is to help you resist persuasion cannot borrow it.

So the register is a **laboratory readout**: quiet surfaces, hairline structure, tabular
figures, and colour used only where it carries meaning. The interface should look like it is
*reporting a measurement*, not selling one.

---

## 2. Tokens

### Colour

| Token | Value | Role |
|---|---|---|
| `--paper` | `#fcfcfd` | page ground, very slightly cool |
| `--card` | `#ffffff` | raised surfaces |
| `--rule` | `#e4e7ec` | hairlines — the main structural device |
| `--ink` | `#10151f` | body text |
| `--graphite` | `#5a6472` | secondary text (5.85:1) |
| `--mist` | `#667080` | captions, eyebrows (4.88:1) |
| `--lens` | `#0e6b70` | the single accent |

**One accent, derived from the subject.** `--lens` is a deep optical teal, taken from the
colour of anti-reflective lens coating. It is not the default SaaS blue and not an acid
accent; it reads clinical rather than promotional, and it appears only on interactive
elements and the wordmark.

**Verdict colours are marks, not fields.** Saturated colour across large areas reads as
alarm, which is its own kind of persuasion. Verdicts appear as small tags with a wash
background:

| Verdict | Colour | Note |
|---|---|---|
| false | `#b42318` | |
| mixed | `#b54708` | |
| true | `#067647` | |
| related only | `#5925dc` | violet — adjacent evidence, deliberately *not* a verdict hue |
| **no match** | `#667085` **grey** | **absence of evidence must never look like a judgement** |

The band label follows the same rule:

| Band | Colour | Note |
|---|---|---|
| Likely trustworthy | `#067647` | only reachable when a fact-check or live source voted |
| Treat with caution | `#b54708` | |
| Low trust | `#b42318` | |
| **Not enough evidence** | `#667085` **grey** | the score is still shown; only the *label* is withheld |

That last row is a trust decision, not an aesthetic one. If "nobody has checked this" were
red, the interface would be inventing a verdict.

### Type

| Role | Face | Why |
|---|---|---|
| UI / body | **Public Sans** | drawn for the US Web Design System — a face designed for public-institution legibility and trust |
| Display | **Instrument Serif** | used with restraint, for the wordmark, the score and section titles |
| Data | system mono stack | tabular figures for similarity, weights, timings — the instrument voice |

Both faces are **self-hosted** (47 KB total, in `dashboard/public/fonts/` and
`extension/fonts/`). A CDN would contradict the product's central promise that it runs
offline with no third-party calls.

The overlay is a special case: `@font-face` is ignored inside a shadow root, so the two
faces are *declared* in the host document under the private names `TruthLens Sans` and
`TruthLens Serif`. Declaring a face renders nothing on its own, so the host page is
unaffected — verified by checking that the page's own `body` font is unchanged after
injection.

---

## 3. The signature: the evidence meter

Most interfaces show a score. This one shows **the score and how much evidence produced it**,
in the same object.

Each segment's width is one engine's *effective weight* — its base weight scaled by its own
confidence. Whatever is left over is drawn hatched and labelled **unmeasured**. Segment
colour shows which way that evidence pushed: red lowered trust, green raised it.

```
19 /100   Low trust

████████████████ ████████ ░░░░░░░░░░░░░░░░░░
fact check 41%   tox 13%  unmeasured 43%
```

The result: **87 backed by 30% evidence looks visibly different from 87 backed by 90%.** The
project's core honesty property — that engines abstain rather than guess — becomes the first
thing you see instead of a field buried in JSON.

The meter carries an `aria-label` stating the measured and unmeasured percentages, and fills
with a 550 ms transition that is disabled under `prefers-reduced-motion`.

---

## 4. Trust principles applied

These are the specific UX decisions made to earn trust rather than assert it.

**1. Provenance is never more than one glance away.** No verdict appears without the claim it
matched, the publisher, the similarity, and a link to the source. The matched claim is quoted
in full so the reader can judge the match themselves.

**2. Uncertainty is shown, not hidden.** The hatched region, the `confidence` figure, and the
per-engine weights all exist so the interface can say "I am not sure" in a way that is
impossible to miss.

**3. Absence is neutral.** `no match` is grey and reads "nobody has checked this yet, not that
it is true" — stated in the colophon and in the engine's own note.

**4. The system shows its work.** Every score expands into a per-factor table with direction,
weight and a sentence of reasoning. Every checked sentence expands into the check-worthiness
signals that selected it. Nothing is asserted without a route to the reasoning.

**5. The interface refuses when it should.** When wording differs from a matched fact-check in
a way that could flip its meaning, the result is a **Verdict withheld** panel that explains
the mismatch — an interface admitting a limitation in the place where a lesser one would show
a confident answer.

**6. Consistent vocabulary across surfaces.** The same five verdict words, the same colours,
the same meter appear in the dashboard, the popup and the overlay. Learning one teaches all
three.

**7. Plain language, active voice.** "Check", not "Submit". "Verdict withheld", not "Warning".
Errors say what happened and what to do: *"Can't reach the analysis service on localhost:8000.
Start it with ./run.sh, then try again."* Empty states invite an action rather than apologise.

**8. Live evidence is drawn as a different kind of thing.** A matched fact-check sits on a
solid rule; live evidence sits in a **dashed** box on a sunken ground, headed "Looked up
live", and never borrows the verdict-tag styling. That is a deliberate visual claim: a
professional's published verdict and a model's reading of an encyclopedia should not look
alike, because they are not equally strong. It also shows both numbers that produced it —
model confidence *and* passage relevance.

**9. The privacy story is stated where it changes.** The masthead switches from "everything
runs on this machine" to "models run here · live lookups query Wikipedia" the moment the
live channel is active, and the colophon names the setting that turns it off.

**10. Status is ambient.** Every surface shows models loaded, corpus size, and "running on this
machine · no API key" — the privacy claim is visible continuously, not buried in a policy.

---

## 5. Accessibility

Checked, not assumed:

- **Contrast** — all text meets WCAG AA. The audit caught `--mist` at 2.99:1; it was darkened
  to 4.88:1. Verdict tags on their washes range 5.20:1–7.02:1.
- **Never colour alone** — every verdict tag carries its word as text, so the meaning survives
  colour-blindness and greyscale printing.
- **Structure** — one `h1`, section rules are real `h2`/`h3` headings, so the result is
  navigable by heading.
- **Keyboard** — visible focus on every control; `Ctrl`/`Cmd + Enter` submits.
- **Forced colours** — the textarea's focus ring falls back to a real outline, since
  `box-shadow` is dropped in forced-colours mode.
- **Motion** — the meter transition and the working pulse both respect
  `prefers-reduced-motion`.
- **Responsive** — no horizontal overflow down to 380 px.

---

## 6. Notes and known compromises

- **Light only.** A single committed appearance, as briefed. There is no dark variant; adding
  one would mean re-validating every contrast pair.
- **The history list is capped at 12** with a "Show all" control, because a long record was
  visually swallowing the page.
- **The overlay uses a 390 px card** fixed to the bottom-right. It does not attempt to anchor
  to the selection, which would fight with page layout and scroll.
- **One CSS bug worth remembering:** a generic `.empty` utility class collided with the
  meter's `.swatch.empty` modifier and, being later in the file, overrode its size. Both were
  renamed to scoped names (`.empty-state`, `.swatch.unmeasured`). Generic class names in a
  shared stylesheet are a trap.
