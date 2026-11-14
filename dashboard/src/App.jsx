import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = "http://localhost:8000";

const BANDS = {
  trustworthy: { color: "var(--true)", label: "Likely trustworthy" },
  caution: { color: "var(--mixed)", label: "Treat with caution" },
  low: { color: "var(--false)", label: "Low trust" },
  // Grey, never a verdict hue: absence of evidence must not look like a judgement.
  insufficient_evidence: { color: "var(--none)", label: "Not enough evidence" },
};
const VERDICT = {
  false: "false",
  mixed: "mixed",
  true: "true",
  related: "related only",
  no_match: "no match",
};
const band = (b) => BANDS[b] ?? { color: "var(--none)", label: b ?? "unknown" };
const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const factorName = (f) => f.replace(/_/g, " ");

/* ------------------------------------------------------------------ *
 * The signature element.
 * Each segment is one engine's EFFECTIVE weight (its base weight scaled
 * by its own confidence). Whatever is left is drawn hollow and labelled
 * "unmeasured", so a score backed by thin evidence looks visibly different
 * from the same score backed by strong evidence.
 * ------------------------------------------------------------------ */
function EvidenceMeter({ explanation }) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, [explanation]);

  const segments = (explanation ?? []).filter((e) => e.weight > 0);
  const used = segments.reduce((a, e) => a + e.weight, 0);
  const unmeasured = Math.max(0, 1 - used);

  return (
    <div>
      <div
        className="meter"
        role="img"
        aria-label={`Evidence backing this score: ${Math.round(used * 100)} percent measured, ${Math.round(unmeasured * 100)} percent unmeasured`}
      >
        {segments.map((e) => (
          <div
            key={e.factor}
            className={`meter-seg ${e.direction}`}
            style={{ width: shown ? `${e.weight * 100}%` : "0%" }}
          />
        ))}
        {unmeasured > 0.001 && <div className="meter-seg unmeasured" />}
      </div>

      <p className="meter-caption">
        Bar width is how much evidence actually backed this score. The hatched part is
        evidence that was <b>not available</b> — usually because no published fact-check
        matches, or a model was unsure.
      </p>

      <div className="meter-key">
        {segments.map((e) => (
          <span className="meter-key-item" key={e.factor}>
            <span className={`swatch ${e.direction}`} />
            {factorName(e.factor)} <b>{Math.round(e.weight * 100)}%</b>
          </span>
        ))}
        {unmeasured > 0.001 && (
          <span className="meter-key-item">
            <span className="swatch unmeasured" />
            unmeasured <b>{Math.round(unmeasured * 100)}%</b>
          </span>
        )}
      </div>
    </div>
  );
}

const LIVE_LABEL = {
  supported: ["Supported by a live source", "var(--true)"],
  contradicted: ["Contradicted by a live source", "var(--false)"],
  inconclusive: ["Live sources were unclear", "var(--none)"],
  no_evidence: ["No live source found", "var(--none)"],
  disabled: ["Live research is off", "var(--none)"],
};

/* Live evidence is visually SEPARATE from a matched fact-check, because it is a different
   kind of thing: a model reading an encyclopedia, not a professional's published verdict.
   Same reason it never borrows the verdict tag styling. */
function LiveEvidence({ live }) {
  const [label, color] = LIVE_LABEL[live.status] ?? [live.status, "var(--none)"];
  const decided = live.status === "supported" || live.status === "contradicted";
  return (
    <div className="live">
      <span className="live-head">
        <span className="live-dot" style={{ background: color }} />
        Looked up live · {label}
      </span>
      {decided && (
        <>
          <p className="evidence-claim">{live.passage}</p>
          <div className="evidence-meta">
            <a href={live.source_url} target="_blank" rel="noreferrer">
              Wikipedia: {live.source_title}
            </a>
            <span className="num">{Math.round((live.confidence ?? 0) * 100)}% model confidence</span>
            <span className="num">{Math.round((live.relevance ?? 0) * 100)}% relevance</span>
          </div>
        </>
      )}
      <p className="live-note">
        {live.note}
        {decided && " Weighted below a published fact-check in the score."}
      </p>
    </div>
  );
}

function ClaimRecord({ claim }) {
  return (
    <li className="claim">
      <div className="claim-head">
        <p className="claim-text">{claim.text}</p>
        <span className={`tag ${claim.verdict}`}>{VERDICT[claim.verdict] ?? claim.verdict}</span>
      </div>

      {claim.matched_claim && (
        <div className="evidence">
          <span className="evidence-label">Matched fact-check</span>
          <p className="evidence-claim">{claim.matched_claim}</p>
          <div className="evidence-meta">
            {claim.publisher && <span>{claim.publisher}</span>}
            {claim.rating && <span>rated {claim.rating}</span>}
            {claim.similarity != null && (
              <span className="num">{Math.round(claim.similarity * 100)}% similar</span>
            )}
            {claim.source_url && (
              <a href={claim.source_url} target="_blank" rel="noreferrer">
                Open source
              </a>
            )}
          </div>
        </div>
      )}

      {claim.live && <LiveEvidence live={claim.live} />}

      {(claim.warnings ?? []).map((w, i) => (
        <div className="guard" key={i}>
          <span className="guard-title">Verdict withheld</span>
          {w}
        </div>
      ))}

      {/* Only show the sentence-long explanation when it adds something the blocks above
          do not already state: the guard box carries the warning text verbatim, and the
          evidence meta already shows publisher, rating and similarity. */}
      {claim.explanation && (claim.warnings ?? []).length === 0 && !claim.matched_claim && (
        <p className="claim-note">{claim.explanation}</p>
      )}

      {(claim.signals ?? []).length > 0 && (
        <details className="reveal">
          <summary>Why this sentence was checked</summary>
          <div className="reveal-body">
            <p style={{ margin: "0 0 6px" }}>
              Check-worthiness {claim.checkworthiness} — a sentence is checked at 0.40 or above.
            </p>
            <ul>
              {claim.signals.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
        </details>
      )}
    </li>
  );
}

function SectionRule({ label, level = "h2" }) {
  const H = level;
  return (
    <div className="section-rule">
      <H className="section-label">{label}</H>
    </div>
  );
}

function Report({ report }) {
  const b = band(report.band);
  const claims = report.fact_check?.claims ?? [];
  const matched = report.fact_check?.claims_matched ?? 0;

  return (
    <div className="verdict-block">
      <div className="readout">
      <div className="verdict-head">
        <div className="score" style={{ color: b.color }}>
          {report.trust_score}
          <span className="score-max"> / 100</span>
        </div>
        <div>
          <div className="verdict-label" style={{ color: b.color }}>
            {b.label}
          </div>
          <div className="verdict-note">
            {claims.length} claim{claims.length === 1 ? "" : "s"} checked · {matched} matched a
            fact-check · {report.timing_ms.total} ms
          </div>
        </div>
      </div>

      <EvidenceMeter explanation={report.explanation} />
      </div>

      {claims.length > 0 && (
        <>
          <SectionRule label="Claims" level="h3" />
          <ul className="claims">
            {claims.map((c, i) => (
              <ClaimRecord claim={c} key={i} />
            ))}
          </ul>
        </>
      )}

      <SectionRule label="How this score was reached" level="h3" />
      <div className="method">
        {(report.explanation ?? []).map((e) => (
          <div className="method-row" key={e.factor}>
            <span className="method-factor">{factorName(e.factor)}</span>
            <span className="method-weight">
              <span className={`dir ${e.direction}`}>{e.direction}</span>
              <br />
              {e.weight}
            </span>
            <span className="method-detail">{e.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);
  const [openId, setOpenId] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [detail, setDetail] = useState(null);
  const resultRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [h, hist] = await Promise.all([
        fetch(`${API}/health`).then((r) => r.json()),
        fetch(`${API}/history?limit=100`).then((r) => r.json()),
      ]);
      setHealth(h);
      setItems(hist.items);
      setTotal(hist.total);
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const check = async () => {
    const value = text.trim();
    if (!value) return;
    setBusy(true);
    setError(null);
    try {
      const resp = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: value }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body?.detail?.[0]?.msg ?? body?.detail?.message ?? `Request failed (${resp.status})`);
      }
      const r = await resp.json();
      setReport(r);
      setOpenId(null);
      refresh();
      requestAnimationFrame(() =>
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      setError(
        String(e.message ?? e) === "Failed to fetch"
          ? "Can't reach the analysis service on localhost:8000. Start it with ./run.sh, then try again."
          : String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const openRow = async (id) => {
    if (openId === id) return setOpenId(null);
    setOpenId(id);
    setDetail(null);
    try {
      setDetail(await fetch(`${API}/report/${id}`).then((r) => r.json()));
    } catch {
      setDetail({ error: true });
    }
  };

  const stats = useMemo(() => {
    if (!items.length) return null;
    const avg = Math.round(items.reduce((a, i) => a + i.trust_score, 0) / items.length);
    const matched = items.filter((i) => i.fact_verdict && !["no_match", "related"].includes(i.fact_verdict)).length;
    const low = items.filter((i) => i.band === "low").length;
    return { avg, matched, low };
  }, [items]);

  const online = Boolean(health);

  return (
    <div className="shell">
      <header className="masthead">
        <h1 className="wordmark">
          <span className="lens-mark" aria-hidden="true" />
          TruthLens
        </h1>
        <div className="telemetry">
          {online ? (
            <>
              <span>
                {/* live_research is a channel, not one of the three analysis models —
                    counting it here would report "4/3 models loaded". */}
                <b>
                  {["toxicity", "ai_detection", "fact_check"].filter(
                    (n) => health.engines?.[n]?.status === "loaded").length}/3
                </b>{" "}
                models loaded
              </span>
              <span>
                <b>{health.corpus_claims.toLocaleString()}</b> fact-checks indexed
              </span>
              <span>
                {health.engines?.live_research?.status === "loaded"
                  ? "models run here · live lookups query Wikipedia"
                  : "everything runs on this machine · no API key"}
              </span>
            </>
          ) : (
            <span className="off">analysis service offline</span>
          )}
        </div>
      </header>

      <main>
        <h2 className="composer-title">Check a claim</h2>
        <p className="composer-sub">
          Paste a sentence, a headline, or a whole paragraph. TruthLens finds the factual claims
          inside it and checks each one against{" "}
          {health?.corpus_claims ? health.corpus_claims.toLocaleString() : "42,277"} published
          fact-checks.
        </p>

        <label htmlFor="claim-input" className="section-label" style={{ display: "none" }}>
          Text to check
        </label>
        <textarea
          id="claim-input"
          className="field"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) check();
          }}
          placeholder="The unemployment rate fell every month last year."
        />
        <div className="composer-actions">
          <button className="btn" onClick={check} disabled={busy || !text.trim()}>
            {busy ? "Checking…" : "Check"}
          </button>
          {busy ? (
            <span className="working">
              <span className="pulse" />
              Three models are reading this on your machine
            </span>
          ) : (
            <span className="hint">
              or press <span className="kbd">Ctrl</span> <span className="kbd">Enter</span>
            </span>
          )}
        </div>

        {error && (
          <div className="notice alert" style={{ marginTop: 18 }} role="alert">
            <span className="notice-title">Couldn't check that</span>
            {error}
          </div>
        )}

        <div ref={resultRef}>
          {report && (
            <>
              <SectionRule label="Result" />
              <Report report={report} />
            </>
          )}
        </div>

        {stats && (
          <>
            <SectionRule label="Record" />
            <div className="stats">
              <div className="stat">
                <div className="stat-value">{total}</div>
                <div className="stat-label">checks run</div>
              </div>
              <div className="stat">
                <div className="stat-value">{stats.avg}</div>
                <div className="stat-label">mean trust</div>
              </div>
              <div className="stat">
                <div className="stat-value">{stats.matched}</div>
                <div className="stat-label">fact-check hits</div>
              </div>
              <div className="stat">
                <div className="stat-value" style={{ color: stats.low ? "var(--false)" : undefined }}>
                  {stats.low}
                </div>
                <div className="stat-label">rated low</div>
              </div>
            </div>
          </>
        )}

        <SectionRule label="History" />
        {items.length === 0 ? (
          <p className="empty-state">Nothing checked yet. Your results will be listed here.</p>
        ) : (
          <ul className="ledger">
            {(showAll ? items : items.slice(0, 12)).map((i) => (
              <li key={i.id}>
                <button
                  className="ledger-row"
                  onClick={() => openRow(i.id)}
                  aria-expanded={openId === i.id}
                >
                  <span className="ledger-score" style={{ color: band(i.band).color }}>
                    {i.trust_score}
                  </span>
                  <span className="ledger-text">{i.text_excerpt}</span>
                  <span className="ledger-meta">
                    {new Date(i.created_at).toLocaleDateString(undefined, {
                      day: "2-digit",
                      month: "short",
                    })}{" "}
                    · {VERDICT[i.fact_verdict] ?? "—"}
                  </span>
                </button>
                {openId === i.id && (
                  <div className="ledger-detail">
                    {!detail && (
                      <span className="working">
                        <span className="pulse" />
                        Loading the full record
                      </span>
                    )}
                    {detail?.error && (
                      <div className="notice alert">That record could not be loaded.</div>
                    )}
                    {detail?.explanation && <Report report={detail} />}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
        {items.length > 12 && !showAll && (
          <button className="more" onClick={() => setShowAll(true)}>
            Show all {items.length} checks
          </button>
        )}

        <p className="colophon">
          <b>How to read this.</b> The meter shows how much evidence backed the score, not just the
          score itself — hatched space is evidence that was never available. A claim with no
          matching fact-check is marked <b>no match</b>, which means nobody has checked it yet, not
          that it is true. When wording differs from a matched fact-check in a way that could flip
          its meaning, the verdict is withheld and the reason is shown.
          <br />
          <br />
          <b>What leaves your machine.</b> All models run here and no API key is used. When a
          claim has no published fact-check, TruthLens can look it up live — that sends search
          terms derived from your text to Wikipedia, and every result that used it is labelled
          “Looked up live”. Set <code>TRUTHLENS_LIVE_RESEARCH=0</code> to keep everything local;
          nothing else changes.
        </p>
      </main>
    </div>
  );
}
