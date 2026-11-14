# TruthLens — Unified AI Moderation & Claim Checker

Highlight text in your browser, click once, and TruthLens **finds the individual claims
inside it** and checks each one across three signals analysed in parallel:

| Vector | What it measures | How |
|---|---|---|
| **AI-generated text** | probability the passage was machine-written | RoBERTa detector, running locally |
| **Toxicity** | probability the passage is offensive / hateful | RoBERTa offensive-language classifier, running locally |
| **Fact check** | whether professional fact-checkers already ruled on each claim | claim extraction + dense retrieval over **42,277** fact-checks from **144 publishers** |
| **Live research** | what a current source says, when no fact-check exists | live Wikipedia retrieval + entailment (NLI) |

**All four models run on your machine.** No API key, no account, no paid tier.

When a claim has **no published fact-check**, TruthLens can research it live: it searches
Wikipedia, retrieves the passages that are actually about the claim, and uses an entailment
model to decide whether they support or contradict it. That is the one feature that leaves
your machine — it sends search terms to Wikipedia, every result that used it is labelled
**"Looked up live"**, and `TRUTHLENS_LIVE_RESEARCH=0` turns it off with nothing else changing.

> **New to the code? Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md)** — a complete walkthrough of
> every file, algorithm and design decision, plus a demo script and expected viva questions.
> **[DESIGN.md](DESIGN.md)** covers the interface: the design system, the evidence meter, the
> trust-UX principles behind it, and the accessibility audit.

---

## Why this is not "three API calls in a trench coat"

1. **Local inference, not hosted inference.** Tokenisation, the forward pass, the softmax and
   the label mapping all happen in `backend/app/engines/`. No third-party inference endpoint
   exists in the request path.
2. **Claim extraction.** A selection is split into sentences and scored for
   *check-worthiness* before anything is retrieved — because the corpus is sentence-indexed,
   and a paragraph embedding averages its sentences into something that resembles none of
   them.
3. **Dense retrieval.** Claims are embedded and L2-normalised so cosine similarity is a
   single matrix–vector product over 42,277 entries — sub-millisecond, no vector database.
   The corpus merges three public datasets, keeping the record with the best provenance
   when they overlap.
4. **Recovered dataset semantics.** LIAR2 ships six numeric labels with no documentation. The
   mapping was *derived*, not assumed: for speakers whose entire credit history is a single
   verdict, the one non-zero credit column must be that row's label. ~100% agreement across
   n=2,283. Reproduce with `scripts/verify_labels.py`.
5. **A measured threshold, not a guessed one.** `scripts/calibrate_threshold.py` runs a real
   experiment — 400 perturbed claims, easy negatives, hard negatives — and reports recall,
   precision and false-match counts across a threshold sweep.
6. **A contradiction guard over the neural score.** Measured: negated claims match their
   original *better* than genuine paraphrases do. No threshold can fix that, so a symbolic
   check sits on top. See below — this is the most important result in the project.
7. **Selective INT8 quantization.** Applied where it is free, rejected where it costs
   accuracy, with the measurement pinned by a test.
8. **Retrieve-then-reason, with relevance gating the reasoning.** Live research pairs
   Wikipedia retrieval with an entailment model — a hybrid neural/symbolic step that can
   see negation, which embeddings cannot. Crucially, a passage must be *relevant* before
   its entailment reading counts (see below).
9. **Concurrent orchestration with failure isolation.** Three engines run simultaneously; one
   dying degrades exactly one vector and the report still returns.

---

## The result that matters most

Sentence embeddings encode **topic**, not **truth conditions**. Measured over 200 corpus
claims that were negated or had a number changed:

> **100%** still retrieved the original as top-1, at mean similarity **0.959** — *higher* than
> genuine paraphrases (0.878).

The dangerous case scores **better** than the good case, so no threshold can separate them.
Here is what that means in practice, on real data:

| | |
|---|---|
| User text | "The measles vaccine causes autism." |
| Best corpus match | *"Science is 'very certain' that the measles vaccine doesn't cause autism."* |
| That claim's rating | **true** |
| Similarity | **0.8601** — above the 0.80 verdict threshold |
| Without the guard | TruthLens reports the anti-vaccine claim as **TRUE** |
| With the guard | `related only`, with "negation mismatch" shown to the user |

The guard compares clause-scoped negation and numeric content between your text and the
matched claim. Scoping matters: *"born in Kenya, **not** the United States"* is contrastive
and **asserts** the claim, while *"did **not** grow up in Kenya"* **reverses** it. Counting
bare `not` tokens conflates the two.

---

## Measured results (this machine, CPU only)

AMD Ryzen 7 5800H (Cezanne), 8 cores / 16 threads, 27 GB RAM, no GPU. Python 3.14.6, torch 2.13.0+cpu.

### Retrieval calibration — `scripts/calibrate_threshold.py`

Re-measured after the corpus grew from 22,928 to 42,277 claims:

| Population | 22,928 claims | 42,277 claims |
|---|---|---|
| top-1 retrieval accuracy | 97.8% | **98.2%** |
| genuine paraphrases | median 0.900 | median **0.904**, p10 0.763 |
| unrelated everyday sentences | max 0.497 | max **0.502** |
| best **non**-matching claim | p90 0.745, p99 0.847 | p90 **0.756**, p99 **0.876** |
| false matches at a 0.80 gate | 2.25% | **6%** |

**Corpus growth cuts both ways.** More claims means better recall, but also a higher chance
that some *other* claim is a confident near-match — so the bar for asserting a verdict had
to rise with the corpus, from 0.80 to **0.85**. A threshold calibrated on 23k claims is
silently wrong at 42k, which is why it is a script you re-run rather than a constant.

Because the "best wrong claim" range overlaps the low end of the positive range, a single
cut-off cannot work. Hence three tiers:

| Similarity | Outcome | Counts toward the score? |
|---|---|---|
| ≥ 0.85 | verdict asserted | yes |
| ≥ 0.62 | `related` — context, explicitly not a verdict | **no** |
| below | `no_match` | no |

**Why 0.85 and not the script's own recommendation.** `calibrate_threshold.py` prints
`RECOMMENDED = 0.90` — the first threshold meeting its stop rule of ≤1% hard false matches.
0.85 is shipped instead, as a stated recall trade:

| Threshold | Recall | Hard false matches | |
|---|---|---|---|
| 0.80 | 0.828 | 24/400 (6%) | too many wrong verdicts |
| **0.85** | **0.723** | **8/400 (2%)** | **shipped** |
| 0.90 | 0.500 | 4/400 (1%) | halves recall to remove 4 cases |

Going from 0.85 to 0.90 removes four hard false matches and costs 31% of all true matches.
A verdict that is missed drops to `related`, which the user still sees along with its
source. A verdict that is wrong puts a stranger's rating on the user's sentence. That is
why the trade goes this way, and why 2% is accepted rather than driven to zero.

One more number from the same run: the hard-negative **maximum is 0.971**. Some pairs of
genuinely different corpus claims sit closer together than any threshold can separate, and
no amount of tuning fixes that. The contradiction guard exists for those.

### Quantization — `scripts/bench.py`

| Engine | fp32 median | int8 median | Speedup | Label agreement | Ships as |
|---|---|---|---|---|---|
| Toxicity | 31.9 ms | 10.9 ms | **2.94×** | **100%** | int8 |
| AI detection | 35.0 ms | 10.7 ms | 3.27× | 80% | **fp32** |

Re-measuring on a loaded machine gives 3.08× / 3.51× for the same two engines: **the ratio
moves with system load, the 100%-vs-80% agreement split does not.** The agreement column is
what the decision rests on, which is why it is the column that is pinned by a test.

INT8 makes the AI detector 3.27× faster but changes one prediction in five (mean |Δp| 0.074,
max 0.118) — enough to push a borderline 0.616 past the 0.70 threshold. A speedup that alters
predictions is not a win, so it ships unquantized, pinned by
`test_quantization_defaults_encode_the_measured_tradeoff`.

### End-to-end

| Metric | Value |
|---|---|
| `/analyze` median, local only | **123 ms** (target: 2000 ms) |
| `/analyze` median, with a live lookup | **251 ms** warm · 2–4 s cold |
| Concurrency speedup | **2.16–2.29×** vs serial |
| Fact-check corpus | **42,277** claims · 144 publishers (22,987 false · 15,707 mixed · 3,583 true) |
| Embedding dimension | 384 |
| Tests | **142**, ~1.2 s, no model downloads |
| API keys required | **0** |

Sample results:

| Input | Trust | Band | Why |
|---|---|---|---|
| "Barack Obama was born in Kenya, not the United States." | 78/100 | **not enough evidence** | matched *"Says Barack Obama grew up in Kenya"* (rated false) at **0.823** — below the 0.85 bar, so the vector abstained and the top band is withheld |
| "The measles vaccine causes autism." | 65/100 | caution | matched a claim rated *true* at **0.8601** — negation guard blocked the verdict |
| "Obama raised taxes 42 times while in office." | 78/100 | **not enough evidence** | matched *"…raised taxes 19 times"* at **0.8457** — numeric guard blocked the verdict |
| "The Eiffel Tower is located in Berlin, Germany." | **34/100** | low | no fact-check; Wikipedia's *Eiffel Tower* lead **contradicts** it (0.9995 × 0.7219 relevance = 0.7215 evidence) |
| "The Eiffel Tower is located in Paris, France." | **91/100** | trustworthy | same passage **supports** it (0.9765 × 0.8209 = 0.8016) |
| "You are a disgusting idiot and nobody wants you here." | **11/100** | low | toxicity 0.9186 |
| "My neighbour repainted his fence pale blue last Tuesday." | 87/100 | **not enough evidence** | clean and human-sounding — but nothing at all is known about whether it is true |

The band is deliberately **not** a pure function of the score: see limitation 6.

---

## Quick start

```bash
./run.sh
```

That is the whole thing. On a fresh clone it creates the virtualenv, installs the
dependencies, builds the fact-check corpus, starts the analysis service and the dashboard,
waits until they are actually healthy, and prints where everything is. Later runs skip
straight to starting (~15 s).

| Command | What it does |
|---|---|
| `./run.sh` | set up anything missing, then start everything |
| `./run.sh --setup` | first-run setup only, no servers |
| `./run.sh --local` | start with live Wikipedia lookups **disabled** |
| `./run.sh --backend` | analysis service only, no dashboard |
| `./run.sh --stop` | stop it |
| `./run.sh --status` | show what is running |
| `./run.sh --check` | run the 142 tests, then check the live service |

First run downloads ~1.6 GB of models (four fp32 checkpoints, ~415 M parameters total) and builds the corpus, so allow 10–15 minutes.
Logs go to `.run/backend.log` and `.run/dashboard.log`.

Then:

* backend — <http://localhost:8000/health>
* dashboard — <http://localhost:5173>
* extension — open `brave://extensions` (or `chrome://extensions`), enable **Developer
  mode**, click **Load unpacked**, select the `extension/` folder.

**Three ways to use it**

1. **Select text → click the toolbar icon.** The popup analyses your selection automatically
   and shows the full breakdown: every claim found, its verdict, the matched fact-check, the
   source link, any contradiction warnings, and why each sentence was considered checkable.
2. **Select text → right-click → *Analyze with TruthLens*.** An overlay appears in the page.
3. **`Ctrl+Shift+T`** does the same as the context menu.

With nothing selected, the popup offers a paste box.

---

## Architecture

```
      Browser extension (MV3)                    React dashboard
      ├── background.js  context menu + fetch    └── history, stats, per-claim detail
      ├── content.js     shadow-DOM overlay
      └── popup.js       auto-analyses selection
                    │                                │
                    └────────── HTTP ────────────────┘
                                 ▼
                    FastAPI gateway  (app/main.py)
                                 │  asyncio.gather → 3 threads
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     ToxicityEngine       AIDetectionEngine     FactCheckEngine
     RoBERTa int8         RoBERTa fp32          claims.py → extract claims
                                                MiniLM   → embed (batched)
                                                cosine   → 42,277-claim index
                                                guard    → negation / numeric check
                                                live     → Wikipedia + entailment (NLI)
            └────────────────────┼────────────────────┘
                                 ▼
                    Aggregator (confidence-weighted fusion, abstention)
                                 ▼
                    SQLite history  →  /history, /report/{id}
```

This preserves the original proposal's architecture (§4: client → gateway → parallel engines
→ aggregator). The engines are local models instead of the Hugging Face Inference API, Google
Fact Check API and Supabase, because the project must run locally and cost nothing.

### Layout

```
truthlens/
├── run.sh                       start backend + dashboard
├── PLAN.md                      the build plan this was executed from
├── HOW_IT_WORKS.md              full technical walkthrough
├── DESIGN.md                    design system, trust UX, accessibility audit
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI gateway, concurrent fan-out
│   │   ├── claims.py            sentence splitting + check-worthiness scoring
│   │   ├── research.py          live Wikipedia retrieval and passage ranking
│   │   ├── aggregator.py        confidence-weighted evidence fusion
│   │   ├── config.py            every tunable, env-overridable
│   │   ├── schemas.py           the wire contract
│   │   ├── store.py             SQLite history
│   │   └── engines/
│   │       ├── base.py          engine protocol, quantization, label resolution
│   │       ├── toxicity.py      offensive-language vector
│   │       ├── ai_detect.py     machine-generated-text vector
│   │       ├── nli.py           entailment: supports / contradicts / neutral
│   │       └── factcheck.py     retrieval, tiering, contradiction guard
│   ├── scripts/
│   │   ├── smoke_models.py      proves all three checkpoints load token-free
│   │   ├── build_corpus.py      downloads LIAR2, embeds it, writes the index
│   │   ├── verify_labels.py     reproduces the label-mapping derivation
│   │   ├── calibrate_threshold.py  the threshold experiment
│   │   └── bench.py             fp32 vs int8, and end-to-end latency
│   └── tests/                   142 tests, no model downloads needed
├── extension/                   MV3: manifest, service worker, overlay, popup, icons, fonts
└── dashboard/                   React + Vite (self-hosted fonts, no CDN)
```

---

## API

`POST /analyze` — `{"text": "...", "url": "optional"}`

```jsonc
{
  "trust_score": 20,
  "band": "low",                  // low | caution | trustworthy
  "confidence": 0.75,             // how much evidence backed the score
  "ai_detection": { "status": "ok", "probability": 0.52, "label": "uncertain", ... },
  "toxicity":     { "status": "ok", "score": 0.25, "flagged": false, ... },
  "fact_check": {
    "status": "ok",
    "verdict": "false",           // worst asserted claim wins
    "claims_found": 2,
    "claims_matched": 1,
    "claims": [
      {
        "text": "Barack Obama was born in Kenya, not the United States.",
        "checkworthiness": 0.70,
        "signals": ["names a person, place or organisation", "uses an assertive verb"],
        "verdict": "false",
        "rating": "false",
        "matched_claim": "Says Barack Obama grew up in Kenya.",
        "similarity": 0.823,
        "publisher": "PolitiFact (via LIAR2 dataset)",
        "source_url": "https://www.politifact.com/search/?q=...",
        "warnings": [],
        "explanation": "Matches a claim rated 'false' by ... at 0.82 similarity."
      },
      {
        "text": "The measles vaccine causes autism.",
        "verdict": "related",
        "warnings": ["negation mismatch: your text has 0 negation word(s), ..."],
        "explanation": "A related fact-checked claim was found, but this is NOT a verdict ..."
      }
    ]
  },
  "explanation": [ { "factor": "fact_check", "direction": "lowers", "weight": 0.41, ... } ],
  "timing_ms": { "toxicity": 71, "ai_detection": 54, "fact_check": 21,
                 "total": 175, "speedup_vs_serial": 2.17 }
}
```

`GET /health` · `GET /history?limit=n` · `GET /report/{id}`

### Optional: live Google Fact Check API

Not required. If a free key happens to be present, the live API is queried too and preferred:

```bash
GOOGLE_FACTCHECK_API_KEY=... ./run.sh
```

Without the variable, that code path is skipped entirely.

### Tunables

| Env var | Default | Effect |
|---|---|---|
| `TRUTHLENS_VERDICT_THRESHOLD` | 0.85 | similarity needed to assert a verdict |
| `TRUTHLENS_LIVE_RESEARCH` | 1 | live Wikipedia lookups; `0` keeps everything local |
| `TRUTHLENS_LIVE_MIN_SIM` | 0.45 | how relevant a passage must be to count as evidence |
| `TRUTHLENS_LIVE_MIN_EVIDENCE` | 0.60 | entailment confidence × relevance |
| `TRUTHLENS_NLI_MODEL` | cross-encoder/nli-deberta-v3-small | the entailment model |
| `TRUTHLENS_RELATED_THRESHOLD` | 0.62 | similarity needed to show a related claim |
| `TRUTHLENS_QUANTIZE_TOXICITY` | 1 | int8 for the toxicity model |
| `TRUTHLENS_QUANTIZE_AI` | 0 | int8 for the AI detector (rejected by measurement) |
| `TRUTHLENS_DEVICE` | cpu | set to `cuda` if you install GPU wheels |

---

## Verification

```bash
cd backend
../.venv/bin/python -m pytest -q                    # 142 tests
../.venv/bin/python scripts/verify_labels.py        # reproduces the label derivation
../.venv/bin/python scripts/calibrate_threshold.py  # reproduces the threshold experiment
../.venv/bin/python scripts/smoke_models.py         # proves checkpoints load with no token
../.venv/bin/python scripts/bench.py --api          # fp32 vs int8 + end-to-end latency
```

The test suite stubs the engines, so it runs in about a second and needs no model download.
Model-dependent behaviour is covered by `smoke_models.py` instead.

**Verified on this machine:** all 142 tests; the label-mapping derivation; the threshold
calibration; all three checkpoints loading with no token; end-to-end `/analyze`; concurrency
at 2.16–2.29×; the dashboard rendering live data; the popup auto-analysing a page selection
and rendering the full per-claim breakdown; and the overlay rendering in a real browser —
including its offline message, its degraded-engine rendering, and that a malicious
`<img onerror=...>` inside a claim is escaped rather than executed.

**Needs one manual step:** the context-menu and `Ctrl+Shift+T` wiring can only run in a real
browser profile, so load the extension unpacked once and try it on a live page.

---

## Honest limitations

Stated plainly, because a moderation tool that oversells itself is worse than no tool. Each
one is visible in the UI, not buried here.

1. **A fact-check database only covers claims fact-checkers already examined.** Most arbitrary
   sentences return `no_match`, which is never presented as "true". Live research softens
   this but does not remove it.
2. **Live research sends data off your machine.** Search terms derived from your text go to
   Wikipedia. It is labelled on every result that used it and disabled with one setting, but
   it is a real change to the privacy story and is stated rather than buried.
3. **A live finding is not a fact-check.** It is an entailment model reading an encyclopedia:
   weaker evidence, capped at 0.55 confidence in the score, kept in separate fields with
   separate wording, and never able to overwrite a published verdict.
4. **Retrieval quality bounds reasoning quality.** An entailment model asked about a bad
   passage still answers confidently. Four failures of this shape have been found and
   fixed. Three were irrelevant passages: a Swedish bibliography line (0.24 evidence), a
   coordinate table (0.62) and a sentence about Obama's half-sister (0.55). Those went away
   once non-prose was filtered out, passages were spread across pages, and the evidence bar
   went up to 0.60.

   The fourth needed a different fix. *"Some parents believe that vaccinations cause
   autism"* was read as entailment at 0.9675 with 0.8346 relevance, so 0.8075 evidence,
   which cleared every gate and pushed a paragraph containing "Vaccines cause autism in
   children" up to 91/100. No relevance threshold rejects that sentence, because it really
   is about the claim, and NLI cannot tell "X believes P" from "P". Attribution verbs are
   now rejected before ranking. The slot that frees up goes to a better sentence on the
   same page ("scientific studies have consistently refuted a causal relationship…") and
   the paragraph scores 77. The failure mode is narrower than it was, not gone.
5. **Live results vary between runs.** Wikipedia's search ranking is not deterministic, so
   the same claim can find different pages on different days and land on a verdict one time
   and abstain another.
6. **Abstention no longer reads as trust, but most text still lands there.** When neither
   the corpus nor live research can settle a claim, only toxicity and AI-detection vote,
   and neither of those measures truth. So the band is not a pure function of the score.
   With no fact-check and no live source, the top band is withheld and the report reads
   `insufficient_evidence` — "Not enough evidence" — in grey, with the reason given in the
   explanation panel. The score is left as it was; only the label changes.

   The cost is that the corpus only covers claims fact-checkers have already looked at, so
   **7 of the 10 sample inputs land in this band**, including true ones like "water boils
   at 100 °C". The tool has nothing to say about them, and now says so.
7. **Embeddings cannot see negation or numeric change.** The contradiction guard catches the
   common cases (auxiliary negation, contractions, differing figures) but it is regex-based,
   not a natural-language-inference model. Paraphrased contradictions with no negation word
   and no number will slip through. The correct next step is an NLI entailment model.
8. **AI-text detection is unreliable**, especially on short or edited passages. "The sun rises
   in the east and sets in the west" scored 0.85 machine-generated. Hence the lowest weight,
   halved confidence under 25 words, and the raw probability always on display.
9. **The models are English-trained.** Non-English input returns scores, but unreliable ones.
   Roughly 60% of the Data Commons rows were dropped for this reason; lifting it means
   switching to a multilingual embedding model and re-running the calibration.
10. **Toxicity ≠ falsehood.** A rude statement can be true. They are combined because the tool
   answers "should I trust this content?", and the explanation panel always separates them.
11. **The corpus still leans US politics**, though far less than before: 144 publishers now,
   including factly.in, dpa-factchecking, PesaCheck, Logically Facts and Vera Files.
12. **Check-worthiness is a heuristic**, not a trained classifier. It errs toward checking too
   much, so harmless sentences sometimes appear with `no_match`.

---

## Credits

* Fact-check corpus, merged from three public datasets:
  * [Data Commons ClaimReview](https://huggingface.co/datasets/SinclairSchneider/datacommons_org_factcheck)
    — ~90k aggregated ClaimReview records with real article URLs
  * [LIAR2](https://huggingface.co/datasets/chengxuphd/liar2) (Xu & Kechadi, IEEE Access 2024) — PolitiFact
  * [Climate-FEVER](https://huggingface.co/datasets/tdiggelm/climate_fever) — climate claims
* Entailment: `cross-encoder/nli-deberta-v3-small`
* Live source: the [Wikipedia API](https://en.wikipedia.org/w/api.php)
* Toxicity: `cardiffnlp/twitter-roberta-base-offensive`
* AI detection: `openai-community/roberta-base-openai-detector`
* Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
