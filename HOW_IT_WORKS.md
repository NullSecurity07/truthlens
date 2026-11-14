# How TruthLens works — every part, in order

This document explains the whole system: what each file does, what each algorithm computes,
why each design decision was made, and where every number in the README came from. It is
written so you can answer any question about the project without having read the code
recently.

**Contents**

1. [The 30-second version](#1-the-30-second-version)
2. [A complete trace of one click](#2-a-complete-trace-of-one-click)
3. [The browser extension](#3-the-browser-extension)
4. [The backend gateway](#4-the-backend-gateway)
5. [Engine 1 — toxicity](#5-engine-1--toxicity)
6. [Engine 2 — AI-generated text](#6-engine-2--ai-generated-text)
7. [Engine 3 — fact check](#7-engine-3--fact-check-the-complicated-one)
8. [The aggregator — how the Trust Score is computed](#8-the-aggregator--how-the-trust-score-is-computed)
9. [Storage and the dashboard](#9-storage-and-the-dashboard)
10. [Where every number came from](#10-where-every-number-came-from)
11. [Every design decision, and why](#11-every-design-decision-and-why)
12. [Demo script and likely questions](#12-demo-script-and-likely-questions)
13. [Glossary](#13-glossary)

---

## 1. The 30-second version

You highlight text in your browser. The extension sends it to a small web server running on
your own machine. That server splits the text into individual factual claims, then runs
three analyses **at the same time**:

| Engine | Question it answers | Technique |
|---|---|---|
| Toxicity | Is this offensive or hateful? | RoBERTa classifier, local |
| AI detection | Was this written by a machine? | RoBERTa classifier, local |
| Fact check | Has a fact-checker already ruled on this? | Sentence embeddings + nearest-neighbour search over 42,277 claims from 144 publishers |

If a claim has no published fact-check, a fourth step can **research it live**: search
Wikipedia and use an entailment model to see whether current sources support or contradict
it (§7.7).

The results are fused into one 0–100 **Trust Score** with an explanation, sent back, and
drawn as an overlay on the page. Every model runs on your machine and no API key exists;
live research is the one step that queries an outside source, and it is labelled wherever
it was used.

---

## 2. A complete trace of one click

Follow one real request all the way through. Input:

> "I went to the shop this morning. The measles vaccine causes autism. Barack Obama was born
> in Kenya, not the United States."

**Step 1 — capture (extension).** You right-click → *Analyze with TruthLens*, or click the
toolbar icon. `background.js` (or `popup.js`) reads the selection and POSTs it to
`http://localhost:8000/analyze`.

**Step 2 — validation (`app/schemas.py`).** Pydantic checks the text is 1–5000 characters.
Empty or oversized input is rejected with HTTP 422 and a message naming the limit.

**Step 3 — fan-out (`app/main.py`).** The gateway launches all three engines concurrently
with `asyncio.gather` over `asyncio.to_thread`. Because PyTorch releases Python's GIL during
a forward pass, these genuinely run in parallel on separate CPU cores.

**Step 4a — toxicity.** The whole text is tokenized and passed through
`cardiffnlp/twitter-roberta-base-offensive`. Output: `offensive` probability **0.25**.

**Step 4b — AI detection.** Same text through
`openai-community/roberta-base-openai-detector`. Output: machine-written probability **0.52**,
label `uncertain`.

**Step 4c — fact check.** This one is multi-stage:

- *Split* into sentences → 3 sentences.
- *Score* each for check-worthiness:
  - "I went to the shop this morning." → 0.27, **dropped** (first-person)
  - "The measles vaccine causes autism." → 0.45, kept
  - "Barack Obama was born in Kenya, not the United States." → 0.70, kept
- *Embed* both surviving claims in one batched forward pass → two 384-dimensional vectors.
- *Search* by multiplying against the 42,277 × 384 corpus matrix → cosine similarity to every
  known fact-checked claim at once.
- *Judge* each:
  - "Barack Obama was born in Kenya…" matches *"Says Barack Obama grew up in Kenya."* at
    **0.823** — below the 0.85 verdict bar, because *born in* and *grew up in* are not the
    same claim → **`related`**, shown as context. It is then **looked up live**, and
    Wikipedia's "Barack Obama" article contradicts it at 0.93 confidence, citing his birth
    at Kapiolani Medical Center in Honolulu.
  - "The measles vaccine causes autism." matches *"Science is 'very certain' that the measles
    vaccine doesn't cause autism."* at **0.775**, rated `true` → the contradiction guard sees
    a negation mismatch → **downgraded to `related`**, no verdict asserted
- *Aggregate*: the worst asserted verdict wins → overall `false`.

**Step 5 — fusion (`app/aggregator.py`).** Each engine contributes a *risk* scaled by its own
*confidence*. Fact-check dominates. Result: **Trust Score 20/100, band `low`**, with three
explanation entries.

**Step 6 — persist (`app/store.py`).** The full report is written to SQLite.

**Step 7 — render.** The extension draws the overlay inside a shadow root. The dashboard
picks the record up on its next poll.

Total elapsed: **123 ms** when everything is answered locally, or **251 ms** warm (2–4 s
cold) when a live lookup runs — about 2.2× faster than running the three engines one after
another.

---

## 3. The browser extension

Manifest V3, loaded unpacked. Four moving parts.

### `manifest.json`

Declares permissions and entry points.

| Field | Why it is there |
|---|---|
| `contextMenus` | to add the right-click item |
| `activeTab` + `scripting` | to read the selection from the current page, only after you click |
| `host_permissions: localhost:8000` | the only network destination allowed |
| `content_scripts` on `<all_urls>` | so the overlay can appear on any page |
| `commands` | the `Ctrl+Shift+T` shortcut |
| `notifications` | fallback when a page forbids overlays |

There is deliberately **no** broad host permission for the internet. The extension can talk
to exactly one address: your own machine.

### `background.js` — the service worker

The coordinator. It has no UI and is woken by events.

- `onInstalled` → creates the context-menu item.
- `contextMenus.onClicked` → takes `info.selectionText`, calls the backend, forwards the
  result to the page.
- `commands.onCommand` → same flow via the keyboard shortcut, reading the selection with
  `chrome.scripting.executeScript`.
- `runtime.onMessage` → serves the popup, which sends text directly.

`callBackend()` handles four failure cases explicitly: nothing selected, oversized selection
(truncated, and the user is told), a non-200 response (the server's own message is shown),
and a network failure (`TruthLens backend offline — start it with ./run.sh`).

`send()` handles the case where no content script exists — restricted pages like
`brave://extensions` or a PDF viewer. It tries to inject the script on demand; if the browser
refuses, it falls back to a system notification so the result is never silently lost.

### `content.js` — the overlay

Runs inside the page. Two design points matter:

**Shadow DOM.** The overlay is attached with `attachShadow({mode:'open'})`. Styles inside a
shadow root cannot be reached by the host page's CSS, and ours cannot leak out. Without this,
any site with aggressive global CSS (`* { font-size: 3em !important }`) would wreck the
overlay, and our styles could break the site.

**No `innerHTML` at all.** Every node is built with `createElement` + `textContent`, because
the matched claims, the live passages and the page's own selected text are all untrusted
input. Verified by a test that feeds `<img src=x onerror="window.__pwned=1">` as a matched
claim and asserts no image element is created and no script runs.

**Bundled typography.** `@font-face` is ignored inside a shadow root, so the two bundled
faces are declared in the host document under private names (`TruthLens Sans`/`Serif`).
Declaring a face renders nothing by itself, so the page is unaffected — checked by asserting
the host page's own `body` font is unchanged.

The overlay shows the score, the **evidence meter**, the per-claim records with their matched
fact-check and any live evidence, and a collapsible **How this score was reached** panel.

### `popup.html` / `popup.js` — the toolbar popup

Opens when you click the toolbar icon. It **auto-analyzes whatever you have selected**:

1. `chrome.tabs.query` finds the active tab.
2. `chrome.scripting.executeScript` runs `window.getSelection().toString()` inside it.
   Clicking the toolbar icon is the user gesture that grants `activeTab`, so no extra
   permission prompt appears.
3. If there is a selection, it analyzes immediately. If not, it shows a paste box.
4. On a restricted page, where scripting is refused, it says so and offers the paste box.

The popup renders more detail than the overlay: the analyzed text, the score ring, all three
vectors, **every extracted claim with its verdict chip, matched claim, source link,
contradiction warnings, and the check-worthiness signals that decided it was worth checking**,
then the full scoring explanation and timing.

All popup rendering uses `createElement` + `textContent`, never `innerHTML`, for the same
injection-safety reason as the overlay.

---

## 4. The backend gateway

`app/main.py`, FastAPI + Uvicorn on port 8000.

### Startup (the `lifespan` handler)

Models load **once**, at startup, not per request — loading takes seconds, inference takes
milliseconds. Each engine is loaded inside its own `try/except`. A model that fails to load
is recorded in `STATE["load_errors"]` and the server **still starts**, reporting itself as
`degraded`. This is why a missing model file produces a clear message at `/health` instead of
a crash halfway through someone's request.

### Concurrency — the part worth understanding

```python
ai, tox, fact = await asyncio.gather(
    _run("ai_detection", text), _run("toxicity", text), _run("fact_check", text))
```

`_run` wraps each engine in `asyncio.to_thread`. This matters:

- `asyncio` alone gives concurrency only for **I/O** waits. Model inference is **CPU** work,
  so pure async would run them one after another.
- Threads normally do not help CPU work in Python because of the GIL — but PyTorch releases
  the GIL inside its C++ kernels. So three threads doing three forward passes genuinely use
  three cores.

The response reports `speedup_vs_serial = (sum of engine times) / (wall-clock total)`.
Measured: **2.16–2.29×**. Not 3× because startup, tokenization and the aggregation step are
not parallel — this is Amdahl's law in miniature, and it is honest to say so.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | the main analysis |
| `GET /health` | per-engine status, label maps, quantization, corpus size |
| `GET /history?limit=n` | past analyses, newest first |
| `GET /report/{id}` | the full stored report |

### Failure isolation

`Engine.run()` in `engines/base.py` never raises. Any exception becomes
`{"status": "unavailable", "error": "<type>: <message>"}` for **that vector only**. The other
two still produce results and the score is computed from what survived. If all three are
dead, the API returns **503** rather than a fabricated score — failing loudly beats inventing
a number.

---

## 5. Engine 1 — toxicity

Model: `cardiffnlp/twitter-roberta-base-offensive`.

### What happens on one call

1. **Tokenize** — text → sub-word integer IDs, truncated at 512 tokens.
2. **Forward pass** — under `torch.inference_mode()` (disables gradient tracking, saving
   memory and time since we never train).
3. **Softmax** — raw logits → probabilities that sum to 1.
4. **Pick the risky class** — see below.
5. **Confidence** — `|p − 0.5| × 2`. A 50/50 output means the model is guessing, so
   confidence is 0; a confident 0.95 gives 0.9.

### Label resolution — the bug that would have inverted everything

This checkpoint labels its classes `non-offensive` and `offensive`. The obvious code is:

```python
if "offensive" in label_name:   # WRONG
```

`"offensive"` is a **substring of** `"non-offensive"`, so depending on dictionary order this
picks the *safe* class as the *risky* one — inverting the entire toxicity vector. Every test
would still pass, the API would still return numbers, and the demo would confidently rate
hate speech as trustworthy.

Hardcoding an index is no safer: this checkpoint is `{0: non-offensive, 1: offensive}` while
the AI detector is `{0: Fake, 1: Real}` — opposite polarity in the same position.

The fix in `engines/base.py`:

1. Match label names against aliases (`offensive`, `toxic`, `hate`, `abusive`, `insult`), but
   **skip any name starting with a negation** (`non`, `not`, `no-`, `neutral`, `clean`,
   `real`, `human`).
2. If nothing matched and exactly one of two labels is negated, take its complement.
3. Otherwise fall back to position and **record that fact** in `label_mapping_source`, which
   `/health` exposes, so a human can see the mapping was guessed.

Eight parametrized tests pin this, including reversed orderings.

### Quantization

At load time, every `nn.Linear` is converted to INT8 with
`torch.ao.quantization.quantize_dynamic`. Weights are stored as 8-bit integers instead of
32-bit floats; activations are quantized on the fly. This is ~4× less memory traffic, and on
CPU memory bandwidth is the bottleneck.

Measured: **31.9 ms → 10.9 ms, a 2.94× speedup, with 100% label agreement.** Free speed, so
it ships enabled.

---

## 6. Engine 2 — AI-generated text

Model: `openai-community/roberta-base-openai-detector`. Mechanically identical to the
toxicity engine — same tokenize → forward → softmax → label-resolve path. Three differences:

**Label polarity.** Labels are `Real` / `Fake`; `Fake` means machine-generated. `real` is in
the negation list, so the alias `fake` resolves correctly regardless of ordering.

**Short-text penalty.** Below 25 words the confidence is **halved**. Detectors of this family
are unreliable on short text, and halving the confidence means the aggregator automatically
gives the vector less influence rather than us pretending otherwise.

**It ships unquantized.** This is the interesting engineering decision. INT8 made it 3.27×
faster — but changed **one prediction in five**:

| | fp32 median | int8 median | Speedup | Label agreement |
|---|---|---|---|---|
| Toxicity | 31.9 ms | 10.9 ms | 2.94× | **100%** |
| AI detection | 35.0 ms | 10.7 ms | 3.27× | **80%** |

Measured drift: mean |Δp| = 0.074, max 0.118 — enough to push a borderline 0.616 across the
0.70 threshold into `likely_ai`. The benchmark's own rule is *a speedup that alters
predictions is not a win*, and this vector is already the least reliable of the three, so
accuracy beats the extra 24 ms. The decision is pinned by a test so nobody re-enables it
without re-reading the measurement.

**Honesty note.** This vector is weighted lowest (0.20), the raw probability is always shown
rather than a bare "AI-written" claim, and the response carries a note saying detection is
probabilistic. In testing, "The sun rises in the east and sets in the west" scored 0.85
machine-generated. That is a real limitation, documented rather than hidden.

---

## 7. Engine 3 — fact check (the complicated one)

Three sub-problems: *which sentences are worth checking*, *which known claim does each one
resemble*, and *is that resemblance trustworthy enough to transfer a verdict*.

### 7.1 The corpus

Built once by `scripts/build_corpus.py` from
[LIAR2](https://huggingface.co/datasets/chengxuphd/liar2) — ~23k statements labelled by
PolitiFact's professional fact-checkers, Apache-2.0, public, no token. After de-duplication:
**42,277 claims from 144 publishers** (22,987 false · 15,707 mixed · 3,583 true), merged
from three public datasets:

| Source | Kept | Why it is there |
|---|---|---|
| Data Commons ClaimReview | 21,461 | ~90k aggregated ClaimReview records carrying the **real article URL** of each fact-check — processed first so it wins de-duplication ties |
| LIAR2 (PolitiFact) | 19,509 | statement-level PolitiFact data; ships no article URL, so its link is a labelled *search* link |
| Climate-FEVER | 1,307 | climate claims, for topical range the other two lack |

Two filters are applied and both are deliberate. **English only** — the embedding, toxicity
and detector models are all English-trained, so a non-English claim would match badly; this
drops ~60% of the Data Commons rows. **Interpretable ratings only** — publishers use 11,782
distinct rating strings, and a verdict we cannot read is a verdict we cannot report, so
1,728 rows were dropped rather than guessed at. The build prints every count.

**The labels are undocumented.** The dataset ships six numeric labels with no key. Rather
than assume the conventional ordering, the mapping was *recovered*:

> For any speaker whose entire credit history contains exactly one verdict, the single
> non-zero credit column must be that row's own label.

Across **n = 2,283** such rows this held at ~100%:

| label | meaning | agreement |
|---|---|---|
| 0 | pants-fire | 258/259 |
| 1 | false | 514/514 |
| 2 | mostly-false | 336/336 |
| 3 | half-true | 442/443 |
| 4 | mostly-true | 383/383 |
| 5 | true | 348/348 |

Reproduce with `scripts/verify_labels.py`. The six ratings are then collapsed to three
verdicts for display; the original rating is preserved in the API response.

Each claim is embedded with `all-MiniLM-L6-v2` into 384 dimensions and **L2-normalised**
(scaled to length 1). Normalising is what lets cosine similarity become a plain dot product
later — a meaningful optimisation, not a formality.

### 7.2 Claim extraction (`app/claims.py`)

**Why split at all?** The corpus is indexed at *sentence* granularity. Embedding a whole
paragraph produces the *average* of its sentences, which resembles none of them. One
checkable falsehood buried in four neutral sentences vanishes below any threshold. This was
the single biggest cause of `no_match`.

**Sentence splitting.** Regex-based, with abbreviations (`Dr.`, `U.S.`, `Jan.`) masked by a
placeholder before splitting and restored after, so they do not create false boundaries.

**Check-worthiness scoring.** Each sentence starts at **0.45** and is adjusted:

| Adjustment | Δ | Rationale |
|---|---|---|
| contains a number | +0.18 | numbers are checkable |
| contains a quantity/unit | +0.12 | "3.2 percent", "$5 million" |
| names a person/place/org | +0.15 | fact-checks are about entities |
| uses an assertive verb | +0.10 | ranking help, not a requirement |
| is hedged ("might", "allegedly") | −0.15 | not an assertion |
| is a question | −0.35 | questions assert nothing |
| reads as opinion | −0.28 | "I think", "beautiful" |
| is first-person | −0.18 | personal chatter |
| is an instruction | −0.20 | "Please click…" |
| fewer than 4 / more than 80 words | −0.30 / −0.20 | fragments and walls of text |

Sentences ≥ **0.40** are checked, best first, capped at 6.

**Why subtract instead of require.** An earlier version *required* a verb from a whitelist and
therefore silently skipped **"Vaccines cause autism in children."** — the exact kind of claim
this tool exists for — because the list contained `causes` but not `cause`. English verbs
cannot be enumerated. Assuming a declarative sentence is checkable and subtracting for
disqualifying features means an unseen verb costs nothing; the worst case becomes "checked a
harmless sentence and reported no match". Four regression tests pin this.

Every signal is returned in the API response, so the UI can show *why* a sentence was or was
not checked.

### 7.3 Retrieval

All surviving claims are embedded in **one batched forward pass**, then:

```python
sims = query_vectors @ corpus_matrix.T     # (n_claims, 22928)
```

Because both sides are L2-normalised, this dot product **is** the cosine similarity. One
matrix multiply compares every claim against all 42,277 entries. `np.argpartition` then finds
the top-k in O(n) instead of sorting all 42,277. No vector database is needed at this scale;
the whole search is well under a millisecond.

### 7.4 Tiering — why one threshold cannot work

`scripts/calibrate_threshold.py` measured, on 400 perturbed corpus claims:

| Population | Similarity |
|---|---|
| genuine paraphrases (positives) | median **0.900**, p10 0.759 |
| unrelated everyday sentences | max **0.497** |
| best **non**-matching corpus claim | mean **0.610**, p90 0.745 |

Top-1 retrieval accuracy was **97.8%**. The problem: the "best wrong claim" distribution
overlaps the low end of the positive distribution. Any single cut-off must either assert
verdicts on wrong claims or reject real paraphrases. So there are three outcomes:

| Similarity | Outcome | Meaning |
|---|---|---|
| ≥ **0.80** | verdict asserted | close enough to inherit the rating |
| ≥ **0.62** | **`related`** | shown as context, explicitly *not* a verdict |
| below | `no_match` | nothing resembles this |

The middle tier is the honest answer to *"close, but I will not put someone else's rating on
your sentence."* Crucially, `related` **contributes zero weight** to the score.

### 7.5 The contradiction guard — the most important part

Sentence embeddings encode **topic**, not **truth conditions**. Measured: taking 200 corpus
claims and *negating* them or *changing a number*:

> **100%** still retrieved the original as top-1, at mean similarity **0.959** — *higher* than
> genuine paraphrases (0.878).

The bad case scores **better** than the good case, so no threshold can fix it. A separate
symbolic check is required.

`contradiction_signals(query, matched)` reports two mismatches:

**Negation mismatch.** Counts clause-scoped negation on both sides. Scoping matters:

| Text | Negation? | Why |
|---|---|---|
| "born in Kenya, **not** the United States" | **no** | contrastive — it *asserts* the claim |
| "did **not** grow up in Kenya" | **yes** | bound to the auxiliary — it *reverses* the claim |

Counting bare `not` tokens conflated these and produced a false trip on a genuine match, so
the pattern requires negation attached to an auxiliary verb (`is/was/did/has… + not`), a
contraction (`didn't`), or an inherently negative word (`never`, `cannot`).

**Numeric mismatch.** If both sides contain numbers and they share none, the rating applies
to a different figure.

Either signal downgrades a verdict to `related` and the reason is shown to the user.

#### The case that justifies all of this

Real data, real numbers:

| | |
|---|---|
| User text | "The measles vaccine causes autism." |
| Best corpus match | *"Science is 'very certain' that the measles vaccine doesn't cause autism."* |
| Corpus rating | **true** |
| Similarity | **0.8601** — above the 0.80 verdict threshold |
| Without the guard | TruthLens reports the anti-vaccine claim as **TRUE** |
| With the guard | `related`, with "negation mismatch" shown |

This is pinned by `test_antivax_claim_never_inherits_the_debunking_claims_true_rating`.

### 7.6 Combining claims

The selection's verdict is its **worst asserted claim** — one debunked sentence in an
otherwise clean paragraph must still pull the score down. `related` and `no_match` claims
never count as matches.

### 7.7 Live research — when nothing has been fact-checked

Most sentences have no published fact-check. Rather than stop at "no match", TruthLens can
look the claim up **now**:

1. **Build a query** (`research.py:search_terms`). Drop stopwords, keep names and numbers.
   A capitalised stopword is kept because it is usually part of a proper name — *except* at
   position 0, where the capital is just sentence case. Without that exception every query
   began with a useless "The".
2. **Search Wikipedia** via the MediaWiki API. No key, no account, descriptive User-Agent.
3. **Split and filter.** Page extracts are split into sentences; section headers
   (`== See also ==`), list fragments and non-English lines are discarded. They are not
   evidence about anything, and an entailment model asked about them still answers.
4. **Rank** every surviving sentence against the claim with the embedding model that is
   *already loaded* — no extra model, no extra download.
5. **Judge** the top passages with an entailment model (`engines/nli.py`):
   premise = the Wikipedia sentence, hypothesis = the user's claim, output = entailment /
   contradiction / neutral.

This is the step retrieval alone can never do. Similarity says two sentences are *about* the
same thing; entailment says whether one *supports* the other. "The measles vaccine does not
cause autism" against "the measles vaccine causes autism" is a **contradiction** at 1.00
confidence — exactly the distinction that defeated the embedding.

#### Relevance gates the reasoning

This is the most important rule in the module, and it exists because of a measured failure.

Checking *"Water boils at 100 degrees Celsius at sea level"* (true), the engine returned
**contradicted**, and the claim scored 23/100. The cause:

| Rank | Relevance | Passage | Entailment |
|---|---|---|---|
| 1 | 0.515 | "His thermometer was calibrated with 0 for the boiling point…" | neutral (correct) |
| 4 | **0.327** | a Swedish bibliography title | **contradiction 0.74** |

The verdict came from the *fourth* passage, barely related, in another language. **An
entailment model asked about an irrelevant premise still answers confidently.** So relevance
must gate the reading, not merely rank it:

```
evidence = entailment confidence x passage relevance
report only if  relevance >= 0.45  AND  evidence >= 0.60
```

Three more failures of the same shape were found while testing, each fixed by a different
guard — which is why the pipeline has four filters rather than one:

| Bad "evidence" found | Evidence score | Guard that now stops it |
|---|---|---|
| Swedish bibliography line | 0.24 | relevance floor |
| Coordinate table on a relevant page | 0.62 | **prose filter** — a table is not an assertion |
| Sentence about Obama's *half-sister* | 0.55 | **evidence bar raised to 0.60** |
| Definitive lead sentence wrongly *rejected* | — | **parenthetical fix** — `[tuʁ ɛfɛl]` in a lead made it look foreign |

And one retrieval fix: a single chatty page was filling every passage slot, crowding out
the lower-ranked article that actually settled the claim, so passages are now **diversified
across pages** (at most two per article).

With that rule the same claim returns *inconclusive* and scores 71.

#### What live research is not

It is **not** a fact-check, and the code, the API and the UI all keep them apart:

| | Published fact-check | Live research |
|---|---|---|
| Source | a professional's verdict | a model reading an encyclopedia |
| Field | `matched_claim`, `verdict` | `live.status`, `live.passage` |
| Wording | false / mixed / true | supported / contradicted / inconclusive |
| Weight | full confidence | capped at **0.55** |
| Can overwrite the other? | — | never |

#### A tuning change that was measured and reverted

Widening the window from 3 pages/4 passages to 4/8 was tried, to catch a decisive sentence
the narrower search had missed. It made things **worse**: checking "Barack Obama was born in
Kenya", it admitted a sentence from the *Auma Obama* article ("patron of the International
Storymoja Festivals in Kenya"), read as a contradiction at 0.98, replacing the correct
citation of Obama's actual birth record. A wider net catches more marginal passages, and a
confident reading of a marginal passage is precisely this design's failure mode. Reverted,
with the reason recorded in `config.py`.

#### Privacy

This is the only part of TruthLens that leaves the machine. Search terms derived from the
selected text go to Wikipedia. Every result that used it is labelled "Looked up live", the
masthead changes to say so, and `TRUTHLENS_LIVE_RESEARCH=0` disables it with nothing else
changing.

### 7.8 The optional Google path

If `GOOGLE_FACTCHECK_API_KEY` exists in the environment, the live Google Fact Check Tools API
is queried too and preferred. Absent the variable, the code path is skipped silently. It is
never required and never prompted for.

---

## 8. The aggregator — how the Trust Score is computed

`app/aggregator.py`. The score is **not** a plain weighted average.

Each engine emits a **risk** in [0,1] and a **confidence** in [0,1]:

| Engine | risk | confidence |
|---|---|---|
| fact check | false → 1.0, mixed → 0.5, true → 0.0 | the match similarity |
| toxicity | the offensive probability | `\|p − 0.5\| × 2` |
| AI detection | the machine-written probability | `\|p − 0.5\| × 2`, halved under 25 words |

Base weights encode a deliberate editorial stance:

| Weight | Value | Why |
|---|---|---|
| fact check | 0.50 | a professional verdict is the strongest evidence about truth |
| toxicity | 0.30 | measured reliably, but says nothing about truth |
| AI detection | 0.20 | least reliable, and machine-written ≠ false |

Then:

```
effective_weight_i = base_weight_i × confidence_i
risk               = Σ(effective_i × risk_i) / Σ(effective_i)
trust_score        = round((1 − risk) × 100)
overall_confidence = Σ(effective_i) / Σ(base_weight_i over engines that reported)
```

Bands: ≥70 `trustworthy`, 40–69 `caution`, <40 `low`.

There is one override on top of that, and it matters. Only the fact-check vector says
anything about truth: toxicity measures tone, and the detector measures authorship. A clean,
human-sounding sentence scores well on both of those while nothing at all is known about
whether it is true.

That is not hypothetical. *"Barack Obama was born in Kenya, not the United States."* came
back as **78/100, "Likely trustworthy", on 0.20 confidence**. Its match sat at 0.823, just
under the 0.85 bar, so the fact-check vector abstained, exactly as it should have, and the
other two carried the score on their own.

So when no fact-check and no live source voted, the top band is withheld and the band
becomes **`insufficient_evidence`**. The score is left exactly as computed and the reason
is added to the explanation array, so nothing about this happens quietly.

### Abstention — the design point worth defending

An engine that has nothing to say contributes **zero weight**, not a neutral 0.5.

If `no_match` scored 0.5, every unmatched claim would silently drag the score toward the
middle, and "we found nothing" would be indistinguishable from "we found something
ambiguous". With abstention, the score stays driven by real evidence and the separate
`confidence` field reports *how much* evidence there was. That is why
"my neighbour repainted his fence" scores 87 with confidence 0.30 — high trust, little
evidence — and both facts are visible.

If **no** engine produced usable evidence, the result is an explicit neutral placeholder of
50 with confidence 0.0, labelled as a placeholder rather than a measurement.

Every engine, including failed and abstaining ones, produces an entry in `explanation` with
its direction, weight and a sentence of reasoning. Nothing is hidden.

---

## 9. Storage and the dashboard

**`app/store.py`** — one SQLite table, `analyses`, created idempotently at startup. Stores
timestamp, excerpt, SHA-256 hash, URL, score, band, the three vector summaries and the full
JSON report. A `threading.Lock` guards the connection because engine threads may write
concurrently. Persistence failure is caught and logged — history is a convenience and must
never fail an analysis. This replaces Supabase: no server, no account, no network.

**`dashboard/`** — React + Vite on port 5173. Polls `/health` and `/history` every 5 seconds,
shows engine status, corpus size, aggregate stats, and an expandable list where each row opens
the full per-claim breakdown, contradiction warnings, check-worthiness signals, source links
and timing.

---

## 10. Where every number came from

Every figure quoted anywhere in this project is reproducible by a command.

| Number | Value | Command |
|---|---|---|
| Corpus size | 42,277 claims, 144 publishers, 384-dim | `scripts/build_corpus.py` |
| Label mapping | ~100% over n=2,283 | `scripts/verify_labels.py` |
| Top-1 retrieval accuracy | 98.2% | `scripts/calibrate_threshold.py` |
| Positive similarity | median 0.904, p10 0.763 | same |
| Unrelated-text max similarity | 0.502 | same |
| Best-wrong-claim similarity | mean 0.621, p90 0.756, p99 0.876 | same |
| False matches at a 0.80 gate | 2.25% at 23k claims -> 6% at 42k | same |
| Entailment latency | 28 ms/pair (int8) | `scripts/smoke_models.py` |
| Mutated claims still matching | 100% at mean 0.959 | same |
| Toxicity quantization | 2.94×, 100% agreement | `scripts/bench.py` |
| AI-detection quantization | 3.27×, 80% agreement | same |
| End-to-end latency | 123 ms local · 251 ms warm live · 2-4 s cold live | `scripts/bench.py --api` |
| Concurrency speedup | 2.16–2.29× | reported in every response |
| Tests | 142 passing, ~1.2 s | `python -m pytest -q` |

The test suite stubs the models, so it runs in about a second and needs no downloads.
Model-dependent behaviour is covered separately by `scripts/smoke_models.py`.

---

## 11. Every design decision, and why

| Decision | Alternative | Why this one |
|---|---|---|
| Local models | Hugging Face Inference API | must be free and run locally; also removes a network dependency from the request path |
| Local corpus + retrieval | Google Fact Check API | needs a key; the local version has the same coverage limitation but no account |
| SQLite | Supabase/Postgres | no server, no account, in the standard library |
| No auth | login + accounts | single local user; an account adds no capability |
| Sentence-level claims | whole-selection embedding | the corpus is sentence-indexed; paragraph vectors dilute the signal |
| Three tiers | one threshold | measured overlap between real matches and best-wrong-matches |
| Contradiction guard | trust the similarity | measured: negated claims match *better* than paraphrases |
| Abstention | neutral 0.5 | prevents silent bias toward the middle |
| Confidence-weighted fusion | fixed weights | lets an unsure engine stop voting instead of adding noise |
| Quantize toxicity only | quantize both / neither | measured 100% vs 80% label agreement |
| Shadow DOM | plain injected div | page CSS cannot break the overlay, and ours cannot break the page |
| `textContent` everywhere | `innerHTML` | claims are attacker-controlled text |
| Blacklist scoring | verb whitelist | whitelists silently drop unseen verbs — it dropped "Vaccines cause autism" |
| Fail loudly (503) | return a default score | a fabricated trust score is worse than an error |

---

## 12. Demo script and likely questions

### A five-minute demo

1. `./run.sh`, show `/health`: three engines loaded, 42,277 claims, int8 on toxicity, and
   the live-research channel reporting `en.wikipedia.org`.
2. Select **"Barack Obama was born in Kenya, not the United States."** → right-click →
   Analyze. The corpus returns **related only** — because *born in* and *grew up in* are not
   the same claim — and then **live research** contradicts it at 0.93, quoting Wikipedia's
   record of his birth in Honolulu, with the article linked. This is the whole pipeline in
   one screen: corpus for context, live source for the verdict.
3. Open **Claims found** to show per-sentence verdicts.
4. Now select **"The measles vaccine causes autism."** → it matches a claim rated **true** at
   0.86 similarity, and the guard blocks the verdict with a visible warning. *This is the
   moment that shows the project understands its own failure modes.*
5. Select an insult → toxicity 0.92, score collapses.
6. Stop the backend, trigger again → clear offline message, not a silent failure.
7. Open the dashboard → history, stats, expandable breakdowns.

### Questions you should expect

**"Isn't this just calling three APIs?"** No API is called. The models run on this machine —
tokenization, forward pass, softmax and label mapping are all in `engines/`. The fact-check
vector is a dense-retrieval system built from a raw dataset, including recovering the
dataset's undocumented label semantics.

**"How do you know the threshold is right?"** It was measured, not chosen. 400 perturbed
claims, easy negatives and hard negatives, sweeping thresholds and reporting recall,
precision and false-match counts. The script is in the repo.

**"What happens when the claim isn't in your database?"** It returns `no_match` and says so.
It never treats absence of a fact-check as evidence of truth. That limitation is inherent to
fact-check databases — the Google API has it too.

**"Can the embedding tell 'X is true' from 'X is false'?"** No, and we measured exactly how
badly: 100% of negated claims still matched their original, at *higher* similarity than real
paraphrases. That is why there is a symbolic contradiction guard on top of the neural score.

**"Why is the AI detector so weak?"** Because detectors of this family are weak on short text.
It is weighted lowest, its confidence is halved under 25 words, the raw probability is always
shown, and the limitation is documented. Over-claiming there would be the real mistake.

**"Why not use a bigger/better model?"** It must run on a laptop CPU in real time with no
paid service. MiniLM at 384 dimensions gives 97.8% top-1 retrieval on this corpus, which is
sufficient; the accuracy bottleneck is corpus coverage and negation handling, not model size.

**"How does live research differ from just calling an API?"** It is a retrieve-then-reason
pipeline built here: query construction, Wikipedia retrieval, passage filtering, embedding-based
ranking, entailment scoring, and a relevance gate derived from a measured false positive. The
only external call is an unauthenticated Wikipedia search.

**"Doesn't live research break your privacy claim?"** It changes it, which is why the claim
changed. The masthead, the colophon, every affected result and the README all say what leaves
the machine, and one setting turns it off.

**"What would you do next?"** Replace the regex contradiction guard with the entailment model
now in the codebase; go multilingual (the corpus loses ~60% of Data Commons rows to the English
filter); and add an approximate nearest-neighbour index if the corpus passes a few hundred
thousand claims, where the exact matrix product stops being sub-millisecond.

---

## 13. Glossary

**Token / tokenizer** — text split into sub-word units the model understands.

**Logits** — raw, unnormalised model outputs, before softmax.

**Softmax** — turns logits into probabilities summing to 1.

**Embedding** — a vector of numbers representing meaning; similar sentences get similar
vectors.

**Cosine similarity** — the angle between two vectors, 1.0 identical, 0.0 unrelated. With
L2-normalised vectors it is exactly the dot product.

**L2 normalisation** — scaling a vector to length 1.

**Dense retrieval** — searching by embedding similarity rather than keyword matching.

**INT8 dynamic quantization** — storing weights as 8-bit integers instead of 32-bit floats,
quantizing activations at runtime. Faster on CPU because less memory is moved.

**GIL** — Python's Global Interpreter Lock, which normally stops threads running Python
bytecode in parallel. PyTorch releases it during native inference, which is why our threads
give real parallelism.

**MV3 / service worker** — Manifest V3, the current Chrome extension format; its background
script is an event-driven worker with no persistent page.

**Shadow DOM** — an isolated DOM subtree whose styles do not interact with the host page.

**Check-worthiness** — how much a sentence looks like a factual claim a fact-checker would
examine.

**NLI / entailment** — a model that reads a (premise, hypothesis) pair and says whether the
premise supports, contradicts, or is neutral toward the hypothesis. Directional, unlike
similarity.

**Retrieve-then-reason** — find candidate evidence, then reason over it. The pattern's
characteristic failure is that the reasoner answers confidently about bad retrieval, which
is why relevance must gate the reasoning.

**ClaimReview** — the schema.org markup fact-checkers publish, which Google Data Commons
aggregates; the source of the corpus's real article URLs.
