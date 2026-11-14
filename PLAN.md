# Plan: TruthLens — local, key-free AI moderation & claim checker

One-line goal: a user highlights any text in their browser, hits one context-menu item, and within ~2 seconds sees a single Trust Score overlay combining AI-generation probability, toxicity, and fact-check retrieval — with every model running on their own machine and no API key, account, or paid service anywhere.

## Classification
Track: **From scratch** — the folder contains only a proposal document and no code (verified: `ran: ls -la` on the project folder). Parked secondary asks: image/OCR and deepfake analysis (named in the proposal §1 and §4, cut from v1 — text-only); Supabase auth; Vercel deployment.

## Interview Ledger
- Q1 deadline/scope → not answered; user replied "do it all in one go all perfect and all free locally running" (exit signal, Stop rule 4).
- Clarification (unprompted): "locally as in the project should run locally" → local = not cloud-deployed; internet at runtime is permitted.
- Total questions spent: **1**. Everything else defaulted and tagged below.

## Goal & Success Criteria
- A user selects text on any web page, right-clicks → "Analyze with TruthLens", and an overlay appears in that page showing a 0–100 Trust Score, an AI-probability percentage, a toxicity flag, and a fact-check verdict with source links.
- The whole system starts with one command and requires **zero** API keys, logins, or paid accounts.
- Median analysis latency ≤ 2000 ms for a 50-word selection on CPU, measured by the timing field the API itself returns.
- `pytest` passes in `backend/`, covering each engine and the aggregator.
- A React dashboard at `localhost:5173` lists past analyses read from the backend.

## Current State
- Only `TruthLens_Project_Proposal_Final_Updated.docx` exists in the project folder; no git repo, no code (verified: `ran: ls -la`).
- Toolchain present: Python 3.14.6, Node v26.1.0, npm 11.16.0, git 2.54.0, Docker 29.7.1, Brave 149.1.91.175 (verified: `ran: python3 --version`, `node --version`, `brave --version`).
- `sqlite3` 3.53.2 is in the Python stdlib (verified: `ran: python3 -c "import sqlite3"`).
- torch 2.13.0 publishes cp314 linux wheels, so local inference works on this interpreter (verified: `ran: curl https://pypi.org/pypi/torch/json`).
- An NVIDIA RTX 3050 Mobile and an AMD Cezanne iGPU are present (verified: `ran: lspci | grep -iE "vga|3d"`).
- Network reaches PyPI (200) and huggingface.co (200); `factchecktools.googleapis.com` answers but returns 403 to unkeyed calls (verified: `ran: curl -o /dev/null -w "%{http_code}"` against each).

## Scope (v1)
Text-only analysis of a user-highlighted selection, through three local engines, aggregated into one score, surfaced in a browser-extension overlay, with a local history dashboard. Everything runs on `localhost`.

## Out of Scope & Parked Items
- **Image OCR and deepfake detection** — proposal §4 mentions "image OCR data"; cut because it adds a fourth model plus an OCR dependency and no part of the demo needs it.
- **Supabase auth and hosted Postgres** — replaced by local SQLite; "runs locally" and "all free" both cut against a hosted account (user).
- **Vercel deployment** — the from-scratch track defers deployment unless the ask includes it; the ask explicitly excludes it (user).
- **Hugging Face Inference API** — requires a token; replaced by the same model families running locally via `transformers`.
- **Chrome Web Store publishing** — the extension is loaded unpacked in developer mode.
- **Multi-user accounts** — one local user; no login screen.

## Approach
Walking skeleton first, through the riskiest unknown. The riskiest unknown is *not* the UI — it is whether three transformer models load and return results fast enough on this CPU with no key. Phase 1 therefore proves exactly that and nothing else; every later phase builds on a measured fact rather than a hope.

The backend is a FastAPI gateway that fans the same text out to three engines concurrently (the proposal's §4 orchestrator, preserved), then aggregates. Each engine implements one small interface, so a failing engine degrades that vector to `unavailable` instead of failing the request. The fact-check vector is a local semantic search over a bundled corpus of previously fact-checked claims — this is the honest local substitute for the Google API, and it returns "no matching fact-check found" rather than inventing a verdict. If a free `${GOOGLE_FACTCHECK_API_KEY}` is ever set in the environment, the same engine additionally queries the live API and merges results; absent the key, that path is skipped silently.

Executor's choice: internal module layout, variable naming, CSS specifics, and the exact React component split.

## Requirements
| # | Requirement | Acceptance check |
|---|---|---|
| R1 | WHEN the backend receives `POST /analyze` with `{"text": "..."}` THE SYSTEM SHALL return a JSON report containing `trust_score`, `ai_detection`, `toxicity`, `fact_check`, and `timing_ms`. | `pytest backend/tests/test_api.py` |
| R2 | WHEN any single engine raises THE SYSTEM SHALL still return 200 with that engine's block marked `{"status": "unavailable", "error": "<message>"}`. | `pytest backend/tests/test_resilience.py` |
| R3 | WHEN the text is empty or longer than 5000 characters THE SYSTEM SHALL return HTTP 422 with a message naming the limit. | `pytest backend/tests/test_api.py` |
| R4 | The three engines SHALL run concurrently, so total latency approximates the slowest engine rather than their sum. | `timing_ms.total` < sum of per-engine timings, asserted in `test_api.py` |
| R5 | THE SYSTEM SHALL require no API key, token, or account to serve a complete analysis. | Fresh clone + `./run.sh` with an empty environment returns a full report |
| R6 | WHEN an analysis completes THE SYSTEM SHALL persist it to SQLite and expose it via `GET /history?limit=n`. | `pytest backend/tests/test_history.py` |
| R7 | WHEN a user selects text and clicks the TruthLens context-menu item THE SYSTEM SHALL inject an overlay into the active page showing the score and the three vectors. | Manual: load unpacked in Brave, select text on any page, observe overlay |
| R8 | WHEN the backend is unreachable THE EXTENSION SHALL show "TruthLens backend offline — start it with ./run.sh" rather than failing silently. | Manual: stop backend, trigger analysis, observe message |
| R9 | THE FACT-CHECK ENGINE SHALL return `no_match` with an explanation when no corpus claim exceeds the similarity threshold, and SHALL NOT emit a true/false verdict in that case. | `pytest backend/tests/test_factcheck.py` |
| R10 | THE DASHBOARD SHALL list past analyses with score, verdict, timestamp, and the analyzed excerpt. | Manual: `npm run dev`, visit `localhost:5173` after running an analysis |
| R11 | THE SYSTEM SHALL report a median `timing_ms.total` ≤ 2000 for a 50-word input on CPU. | `python backend/scripts/bench.py` prints the median |
| R12 | WHEN a model file is missing from the local cache and the network is down THE SYSTEM SHALL report which model failed to load at startup rather than crashing mid-request. | `pytest backend/tests/test_resilience.py` |

## Key Decisions
- **Inference location**: local `transformers` pipelines, not any hosted inference API — (user: "all free", "locally running").
- **Toxicity model**: `cardiffnlp/twitter-roberta-base-offensive` — [assumed: this public RoBERTa checkpoint exists on the Hub and needs no token; matches the proposal's "RoBERTa optimized for offensive language" — if wrong: fall back to `unitary/toxic-bert`, verified in Phase 1 by actually downloading it].
- **AI-text model**: `openai-community/roberta-base-openai-detector` — [assumed: public, token-free, RoBERTa-based — if wrong: fall back to `Hello-SimpleAI/chatgpt-detector-roberta`, verified in Phase 1].
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` — [assumed: public and small enough for CPU — if wrong: fall back to `sentence-transformers/paraphrase-MiniLM-L3-v2`, verified in Phase 3].
- **Compute device**: CPU wheels (`--index-url .../whl/cpu`) — [assumed: three small models clear the 2 s budget on CPU — if wrong: an RTX 3050 is present (verified: `lspci`); reinstall CUDA wheels and set `TRUTHLENS_DEVICE=cuda`].
- **Storage**: SQLite via the stdlib — no server, no account, zero install cost (verified: `sqlite3` importable).
- **Auth**: none. Single local user; the dashboard binds to localhost only.
- **Frontend build**: React via Vite — [assumed: `npm create vite` scaffolds a React+TS app offline-capable after first install — if wrong: hand-write a single-file React app served by the backend].
- **Fact-check source**: bundled local JSONL corpus + semantic retrieval, with the free Google API as an optional keyed enhancement — the only design that satisfies "free", "local", and "returns real sources" at once.

## Data & State Changes
Single SQLite file at `backend/data/truthlens.db`, created on first run. One table:

```sql
CREATE TABLE IF NOT EXISTS analyses (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT    NOT NULL,          -- ISO-8601 UTC
  text_excerpt TEXT    NOT NULL,          -- first 280 chars of input
  text_hash    TEXT    NOT NULL,          -- sha256 of full input, for dedupe
  trust_score  INTEGER NOT NULL,
  ai_probability REAL,
  toxicity_score REAL,
  fact_verdict TEXT,                      -- true|false|mixed|no_match|unavailable
  source_url   TEXT,
  report_json  TEXT    NOT NULL           -- full response, for the dashboard
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
```

Migration: none needed — the table is created idempotently at startup. Rollback: delete the `.db` file; it holds only local history and is regenerable.

The fact-check corpus lives at `backend/data/claims.jsonl` (one claim per line: `claim`, `verdict`, `source`, `publisher`), with embeddings cached to `backend/data/claims_emb.npy`. Both are regenerable by `python backend/scripts/build_corpus.py`.

## Interfaces, Integrations & Credentials

`POST /analyze` → request `{"text": "<1..5000 chars>", "url": "<optional page url>"}`

```json
{
  "trust_score": 42,
  "band": "caution",
  "ai_detection": {"status":"ok","probability":0.83,"label":"likely_ai","model":"..."},
  "toxicity":     {"status":"ok","score":0.12,"flagged":false,"label":"non-offensive","model":"..."},
  "fact_check":   {"status":"ok","verdict":"false","matched_claim":"...","similarity":0.81,
                   "publisher":"...","source_url":"https://...","alternatives":[]},
  "timing_ms": {"ai_detection":410,"toxicity":380,"fact_check":95,"total":455},
  "text_hash": "sha256:...",
  "analyzed_at": "2026-08-18T10:04:11Z"
}
```

`GET /health` → `{"status":"ok","models":{"ai_detection":"loaded","toxicity":"loaded","fact_check":"loaded"},"device":"cpu"}`
`GET /history?limit=50` → `{"items":[{…row…}]}`

- **External deps at runtime**: none required. One-time model download from `huggingface.co` at first start (public repos, no token).
- **Optional integration**: Google Fact Check Tools API, called only when `${GOOGLE_FACTCHECK_API_KEY}` is present in the environment. Never required; never prompted for.
- **Secrets**: none needed for v1. The one optional key is read from the environment as `${GOOGLE_FACTCHECK_API_KEY}` and is never logged or persisted.
- **Frozen surface**: the `/analyze` response shape above is consumed by both the extension and the dashboard — changing a field name means changing all three.

## Edge Cases & Failure Handling
| Case | Behavior |
|---|---|
| Empty / whitespace-only selection | HTTP 422, "Provide between 1 and 5000 characters of text." |
| Selection > 5000 chars | HTTP 422 naming the limit; the extension truncates and tells the user it truncated. |
| One model fails to load at startup | Server still starts; `/health` reports that engine as `failed`; `/analyze` returns its block as `unavailable` with the reason. |
| All three engines unavailable | HTTP 503 with the three reasons — fail loudly, never a fake score. |
| No corpus claim above the similarity threshold | `verdict: "no_match"` plus the nearest claim shown as context, explicitly labelled "not a verdict". |
| Backend offline when the extension fires | Overlay shows "TruthLens backend offline — start it with ./run.sh". |
| Model download interrupted on first run | Startup aborts with the model id and the retry command, rather than half-loading. |
| Non-English text | Scores still return; the overlay carries a "models are English-trained" note [A6]. |
| Duplicate analysis of identical text | Still analyzed and stored; `text_hash` lets the dashboard collapse repeats. |

## Risks, Landmines & Adaptations
- **"All free" + "runs locally" (user)** → every hosted dependency in the proposal (HF Inference API, Google Fact Check API, Supabase, Vercel) is replaced by a local equivalent; the plan gains a bundled corpus and a SQLite store, and loses the auth and deploy phases entirely.
- **Credentials the idea silently needs** (from-scratch signature landmine) → resolved to *zero* required credentials; R5 exists specifically to keep it that way, and the one optional key degrades to a skipped code path.
- **The Google Fact Check API only matches claims that professional fact-checkers already covered** — an arbitrary highlighted sentence usually has no match. The local corpus inherits this exactly. → Adapted by making `no_match` a first-class, honestly-labelled outcome (R9) instead of a failure or a fabricated verdict. This is the single most important honesty property of the build.
- **AI-text detectors are unreliable on short or edited text** → the overlay labels the AI vector as an estimate and shows the raw probability, never a bare "AI-written" claim; the trust score weights it lowest of the three.
- **Python 3.14 is new enough that a dependency may lack wheels** → torch verified (cp314 wheels exist); the remaining installs are checked in Phase 1 before any code depends on them.
- **CPU inference may miss the 2 s target** → Phase 1 measures it before the UI is built; an RTX 3050 is present as the documented escape hatch.
- **Residual risk: model checkpoint names are training-derived, not verified** → Phase 1 downloads each one and falls back to a named alternative on failure; no later phase assumes a specific checkpoint loaded.

## Assumptions Ledger
| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|-----------|-------|----------------------|-------|
| A1 | `cardiffnlp/twitter-roberta-base-offensive` is public and token-free | proposal §3 names RoBERTa for offensive language | toxicity engine swaps checkpoint; no API change | Phase 1 downloads it; falls back to `unitary/toxic-bert` |
| A2 | `openai-community/roberta-base-openai-detector` is public and token-free | proposal §3 names RoBERTa for detection | AI engine swaps checkpoint; no API change | Phase 1 downloads it; falls back to `Hello-SimpleAI/chatgpt-detector-roberta` |
| A3 | `sentence-transformers/all-MiniLM-L6-v2` is public and CPU-viable | needed for local semantic retrieval | fact-check engine swaps to a smaller MiniLM | Phase 3 downloads it |
| A4 | CPU inference meets the ≤2 s median for 50 words | three small models, run concurrently | reinstall CUDA wheels, set `TRUTHLENS_DEVICE=cuda` | Phase 1 `bench.py` prints the median |
| A5 | "All in one go" means build it now, not just plan it | user's words plus "develop this project" | if wrong, the plan alone is still the deliverable | User sees the built tree and says otherwise |
| A6 | English-only inputs are acceptable for the demo | the named checkpoints are English-trained | non-English scores are unreliable | Overlay carries a visible note |
| A7 | Loading an unpacked MV3 extension in Brave is acceptable for the demo | no store account exists and none is free-and-instant | if a store listing is required, add a packaging phase | README documents the load steps |
| A8 | A bundled fact-check corpus of a few thousand claims is enough to demo the vector | full coverage is impossible offline anyway | more `no_match` results than a keyed API would give | Phase 3 prints the corpus size |
| A9 | The team wants the proposal's own architecture preserved, not replaced | the docx is the graded artifact | a different architecture could fail evaluation | Gateway + 3 parallel engines + aggregator mirrors proposal §4 |

## Open Items (none blocking)
- Google Fact Check API key — proceed without one; the code path activates automatically if `${GOOGLE_FACTCHECK_API_KEY}` appears in the environment.
- Firefox support — proceed with Chromium/Brave MV3 only; the manifest is close to Firefox-compatible but untested.
- GPU acceleration — proceed on CPU; `TRUTHLENS_DEVICE=cuda` is wired but unexercised.

## Verification
- `cd truthlens/backend && ../.venv/bin/python -m pytest -q` → all tests pass.
- `curl -s localhost:8000/health | python -m json.tool` → all three engines `loaded`.
- `curl -s -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d '{"text":"Vaccines cause autism."}' | python -m json.tool` → a report with all three vectors populated.
- `../.venv/bin/python scripts/bench.py` → prints median total latency; R11 requires ≤ 2000 ms.
- Human confirmation: start `./run.sh`, load `extension/` unpacked in Brave via `brave://extensions` → Developer mode → Load unpacked, highlight a sentence on any news site, right-click → "Analyze with TruthLens", see the overlay. Then open `localhost:5173` and see that analysis listed.

## Build Phases

- [x] Phase 1: Prove local inference on this machine (walking skeleton through the riskiest unknown)
      Done when: `../.venv/bin/python scripts/smoke_models.py` prints a label and score from each of the three checkpoints plus a per-model load and inference time, with no token set.
      Steps:
      - Install `fastapi uvicorn[standard] transformers sentence-transformers numpy pytest httpx` into the existing venv.
      - Write `scripts/smoke_models.py` that loads each checkpoint and runs one inference.
      - Run it; on any 404/gated error, switch that checkpoint to its A1/A2/A3 fallback and record the swap in the ledger.
      - Record measured load and inference milliseconds in the README.
      Covers: R5, R12; checks: A1, A2, A3, A4

- [x] Phase 2: Backend skeleton with health and the engine interface
      Done when: `curl localhost:8000/health` returns 200 with each engine's load status, and `pytest tests/test_health.py` passes.
      Steps:
      - `app/config.py` (model ids, thresholds, device from `TRUTHLENS_DEVICE`), `app/schemas.py` (pydantic request/response), `app/engines/base.py` (engine protocol + `EngineResult`).
      - `app/main.py` with lifespan-based model loading that records per-engine load status instead of crashing.
      - Test asserting a failed engine leaves the server healthy-but-degraded.
      Covers: R12

- [x] Phase 3: Toxicity and AI-detection engines
      Done when: `pytest tests/test_engines.py` passes, asserting an offensive sample scores above the flag threshold and a neutral sample below it.
      Steps:
      - Implement both engines against the Phase 1 checkpoints, each returning `EngineResult`.
      - Wrap every engine call so an exception becomes `status: unavailable` with the message (R2).
      - Tests use a fixed offensive and a fixed neutral string, asserting relative ordering rather than absolute values.
      Covers: R2

- [x] Phase 4: Local fact-check engine and corpus
      Done when: `pytest tests/test_factcheck.py` passes, including a case that must return `no_match`, and `scripts/build_corpus.py` prints the corpus size.
      Steps:
      - `scripts/build_corpus.py` writes `data/claims.jsonl` and caches embeddings to `data/claims_emb.npy`.
      - Engine embeds the query, cosine-ranks the corpus, applies the similarity threshold, returns verdict or `no_match` with the nearest claim as labelled context.
      - Add the optional Google Fact Check path, guarded by `${GOOGLE_FACTCHECK_API_KEY}` and skipped when absent.
      Covers: R9; checks: A3, A8

- [x] Phase 5: Concurrent orchestration, trust score, and persistence
      Done when: `pytest -q` passes the whole suite including R1, R3, R4, R6, and `curl -X POST /analyze` returns a full report that then appears in `GET /history`.
      Steps:
      - `app/aggregator.py` computes the 0–100 score and band from the three vectors, with the weights documented in the file.
      - `app/main.py` fans out to the three engines with `asyncio.gather` over threads; assert total < sum of parts (R4).
      - `app/store.py` writes the row and serves `GET /history`.
      - Input validation for empty and >5000-char text (R3).
      Covers: R1, R3, R4, R6

- [x] Phase 6: Browser extension (MV3)
      Done when: loaded unpacked in Brave, selecting text and using the context menu renders the overlay with all three vectors; stopping the backend shows the offline message.
      Steps:
      - `manifest.json` (MV3, `contextMenus`, `scripting`, `activeTab`, host permission for `localhost:8000`).
      - `background.js` creates the context-menu item and POSTs the selection.
      - `content.js` + `overlay.css` inject a shadow-DOM overlay so page styles cannot leak in.
      - Handle the offline and truncation cases explicitly (R8).
      Covers: R7, R8; checks: A7

- [x] Phase 7: Dashboard, docs, and end-to-end verification
      Done when: `npm run dev` serves a dashboard at `localhost:5173` listing history from the backend, `scripts/bench.py` prints a median ≤ 2000 ms, and `README.md` documents every command in the Verification section.
      Steps:
      - Vite React app with a history table and a paste-text-to-analyze box.
      - `run.sh` starting backend and dashboard together.
      - `scripts/bench.py` measuring the median over 10 runs (R11).
      - README with setup, run, extension-load steps, measured latencies, and the honest limitations of each vector.
      Covers: R10, R11; checks: A4, A6


---

## Execution record (all phases complete)

| Phase | Evidence |
|---|---|
| 1 Prove local inference | `scripts/smoke_models.py` exit 0. All three PRIMARY checkpoints resolved, no fallback needed: toxicity 66 ms, AI 55 ms, embedding 13 ms inference. A1/A2/A3 confirmed. |
| 2 Backend skeleton + health | `/health` returns 200 with per-engine load status, label maps and quantization notes. |
| 3 Toxicity + AI engines | Both load and score; label resolution verified live (`non-offensive`/`offensive` and `Fake`/`Real` both mapped correctly). |
| 4 Fact-check engine + corpus | 22,928 claims embedded (384-dim). "Barack Obama was born in Kenya" matched "Says Barack Obama grew up in Kenya" at 0.82 similarity, verdict false. |
| 5 Orchestration + fusion + storage | 39 tests pass (incl. a resilience suite for R2 and R12). Concurrency measured at 2.06–2.18× vs serial. History and `/report/{id}` verified. |
| 6 Browser extension | Overlay rendered in a real browser with a real API response: shadow root isolated, score 25, 3 vector rows, quoted claim, PolitiFact link, 3 explanation items. Offline path, degraded-engine path and HTML-escaping all verified. |
| 7 Dashboard + docs + bench | Dashboard renders live health, stats and expandable history (verified via Playwright). Benchmark: end-to-end median 104 ms vs the 2000 ms target. |

### Assumptions resolved

| ID | Outcome |
|----|---------|
| A1 | CONFIRMED — `cardiffnlp/twitter-roberta-base-offensive` loaded token-free |
| A2 | CONFIRMED — `openai-community/roberta-base-openai-detector` loaded token-free |
| A3 | CONFIRMED — `sentence-transformers/all-MiniLM-L6-v2` loaded token-free |
| A4 | CONFIRMED and exceeded — 104 ms median, not the feared ~2000 ms. GPU never needed. |
| A5 | Held — the project was built, not just planned |
| A6 | Unchanged — English-only, documented in README limitations |
| A7 | Unchanged — unpacked load documented; icons now generated so the toolbar entry is real |
| A8 | Exceeded — 22,928 claims, far more than the "few thousand" assumed |
| A9 | Held — the proposal's gateway + 3 parallel engines + aggregator is preserved exactly |

### Deviation from plan, with reason

**Per-engine quantization** was not in the plan; the benchmark forced it. INT8 gave the
toxicity model 2.94x at 100% label agreement, but the AI detector 3.27x at only 80%
agreement (mean |dp| 0.074). The plan's own rule - "a speedup that alters predictions is
not a win" - meant shipping the AI detector unquantized. The decision is pinned by
`test_quantization_defaults_encode_the_measured_tradeoff`.

### Not verified

The context-menu and keyboard-shortcut wiring in `background.js` requires the extension to
be loaded unpacked in a real browser profile, which cannot be done headlessly. The overlay
it drives IS verified. Loading it is the one manual step: `brave://extensions` -> Developer
mode -> Load unpacked -> select `extension/`.

---

## Iteration 2 — per-claim checking (added after user feedback)

Feedback: *"when i select text and click on extension it should auto show the claims / the
popup extension should show more detailed results / make all claims better."*

- [x] Phase 8: Extract individual claims from a selection
      Done when: `pytest tests/test_claims.py` passes and `/analyze` returns a per-claim
      breakdown with check-worthiness scores and the signals behind them.
      Steps:
      - `app/claims.py`: sentence segmentation with abbreviation masking; check-worthiness
        scoring; cap at 6 claims; never return empty for non-empty input.
      - `engines/factcheck.py`: batch-embed all claims in one forward pass, retrieve each,
        aggregate with worst-verdict-wins.
      Covers: the "show the claims" request

- [x] Phase 9: Calibrate the similarity threshold from data
      Done when: `scripts/calibrate_threshold.py` prints the positive/negative distributions
      and a threshold sweep, and `config.py` cites the measured values.
      Steps:
      - 400 perturbed corpus claims as positives; everyday sentences as easy negatives;
        best non-matching corpus claim as hard negatives; mutated claims as the danger set.
      - Replace the guessed 0.55 with a measured two-tier scheme.
      Covers: "make all claims better"

- [x] Phase 10: Contradiction guard
      Done when: `test_antivax_claim_never_inherits_the_debunking_claims_true_rating` passes.
      Steps:
      - Clause-scoped negation detection and numeric comparison between query and match.
      - Downgrade a verdict to `related` when either mismatches, with the reason shown.

- [x] Phase 11: Popup auto-analysis and detailed results
      Done when: opening the popup with text selected analyses it without any click and
      renders per-claim verdicts, matched claims, sources, warnings and signals.
      Steps:
      - `popup.js` reads the selection via `chrome.scripting` under `activeTab`.
      - Full detail rendering, all via `createElement`/`textContent`.
      - Overlay and dashboard gain the same per-claim panel.

- [x] Phase 12: Document everything
      Done when: `HOW_IT_WORKS.md` explains every file, algorithm, number and decision.

### What the measurements changed

| Finding | Consequence |
|---|---|
| Paragraph embeddings dilute the signal | claims are now extracted and matched per sentence |
| Best-wrong-claim similarity overlaps real paraphrases | single threshold replaced by verdict (0.80) / related (0.62) tiers |
| Negated and number-changed claims match at 0.959, HIGHER than real paraphrases (0.878) | symbolic contradiction guard added; no threshold could have fixed this |
| Bare-token negation counting misfires on contrastive "X, not Y" | negation scoped to auxiliary verbs |
| A verb whitelist silently dropped "Vaccines cause autism in children." | scorer inverted to a blacklist; 4 regression tests added |

### Test count

39 → **89** passing.

### Still not verified

Unchanged from iteration 1: the context-menu and keyboard-shortcut wiring needs a real
browser profile. Everything they invoke — the popup flow, the overlay, the offline path, the
degraded path, HTML escaping — is verified in a real browser with the Chrome APIs shimmed.

---

## Iteration 3 — bigger corpus + live research

Feedback: *"now make the corpus bigger and live where it can research too."*

- [x] Phase 13: Merge three public datasets into one corpus
      Done when: `scripts/build_corpus.py` prints the per-source counts and the index rebuilds.
      Result: **22,928 → 42,277 claims, 1 → 144 publishers.** Data Commons ClaimReview
      (21,461) contributes real article URLs and wins de-duplication ties over LIAR2's
      search links; LIAR2 keeps 19,509; Climate-FEVER adds 1,307.
      Two deliberate filters, both counted in the build output: English-only (the models are
      English-trained) and interpretable-ratings-only (11,782 distinct rating strings; a
      verdict we cannot read is a verdict we cannot report).

- [x] Phase 14: Recalibrate on the larger corpus
      Done when: `scripts/calibrate_threshold.py` reruns and `config.py` cites the new numbers.
      Result: top-1 accuracy 97.8% → **98.2%**, but false matches at the 0.80 gate went
      **2.25% → 6%**, because more claims means more chances that some *other* claim is a
      confident near-match. The verdict bar therefore rose to **0.85**.

- [x] Phase 15: Live research (Wikipedia retrieval + entailment)
      Done when: a claim with no published fact-check returns a live finding citing a real
      Wikipedia article, and `pytest tests/test_research.py` passes.
      Steps:
      - `app/research.py`: query construction, MediaWiki search, passage filtering, ranking.
      - `app/engines/nli.py`: entailment engine, labels resolved by name across checkpoints.
      - Wired into the fact-check engine as a strictly weaker channel, capped at 0.55
        confidence, never able to overwrite a published verdict.

- [x] Phase 16: Surface it honestly in all three interfaces
      Done when: live evidence renders in dashboard, popup and overlay in a visually distinct
      form, and the privacy story is stated wherever it changes.

### What the measurements changed (again)

| Finding | Consequence |
|---|---|
| Corpus growth raised false matches 2.25% → 6% at the old gate | verdict threshold 0.80 → 0.85 |
| A 0.33-relevance bibliography line was read as a contradiction at 0.74, scoring a TRUE claim 23/100 | relevance now **gates** the entailment reading: `evidence = confidence × relevance`, with floors at 0.45 and 0.40 |
| Widening the evidence window to 4 pages / 8 passages | **reverted** — it cited the *Auma Obama* article for a claim about Barack Obama at 0.98 confidence |
| fp16 checkpoints crash dynamic quantization | cast to fp32 before quantizing, in both engine base classes |
| Live research loaded a 140 MB model inside every test | toggle read at call time; suite went 107 s → 1.2 s |

### Test count

89 → **119** passing.

### Deliberately not done

**Multilingual.** The English filter drops ~60% of the Data Commons rows. Lifting it means
switching to a multilingual embedding model and re-running the whole calibration, and would
leave the toxicity and detector vectors English-only — a mixed story rather than a finished
one. Recorded as the top item of future work instead of half-built.

---

## Iteration 4 — launcher, and the live-research hardening it exposed

Feedback: *"make a shell script to run/start it."*

- [x] Phase 17: A launcher that sets up, starts, verifies and stops
      `./run.sh` with `--setup`, `--backend`, `--local`, `--stop`, `--status`, `--check`.
      It creates the venv, installs deps, builds the corpus, waits for real health rather
      than assuming, and prints where everything is.

### Bugs the launcher testing surfaced (in the script)

| Bug | Fix |
|---|---|
| `local a="$1" b="$RUNDIR/$a.pid"` — bash expands all words before `local` runs, so `$a` was unset under `set -u` | split into two statements |
| The EXIT trap tore down the *already running* instance when a second start was refused | teardown only touches pids this invocation started |
| `kill -0 "$(cat pid \|\| echo 0)"` — `kill -0 0` signals the caller's own process group and always succeeds | never fall back to pid 0 |
| `--stop` from another terminal made the first terminal report a crash | a missing pid file is a deliberate stop, not a crash |

### Bugs the repeated live lookups surfaced (in the product)

| Failure | Evidence score | Fix |
|---|---|---|
| Coordinate table read as entailment at 0.92, scoring "the Eiffel Tower is in Berlin" **91/100 trustworthy** | 0.62 | prose filter: reject tables, coordinate blobs and digit-dense text |
| One chatty page filled all four passage slots, crowding out the article that settles the claim | — | diversify: at most two passages per page |
| The Eiffel Tower lead sentence was **rejected as foreign** because of its IPA gloss `[tuʁ ɛfɛl]` | — | measure the language ratio with parenthetical glosses stripped |
| A sentence about Obama's half-sister cited as the source for a claim about Barack Obama | 0.55 | evidence bar 0.40 → 0.60, set from measured cases |

Every live-research failure so far has had the same shape: **a confident reading of an
insufficiently relevant passage.** Relevance, prose-ness and an evidence floor are three
different guards against one failure mode.

### Test count

119 → **129** passing.

### Known, not fixed

**Abstention raises the band.** When nothing can settle a claim, only toxicity and
AI-detection vote, so a false claim can read "likely trustworthy" on ~25% evidence. The
meter and the confidence figure disclose this, but the band label ignores confidence.
The fix is to make the band a function of score AND confidence — deliberately left for a
decision rather than changed unilaterally.
