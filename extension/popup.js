/* TruthLens popup.
 *
 * Opens, reads whatever is selected on the page, and analyses it immediately —
 * clicking the toolbar icon is the user gesture that grants `activeTab`, so no
 * extra permission prompt appears. With no selection it offers a paste box.
 *
 * Every value that came from a web page or from the dataset is rendered with
 * createElement/textContent, never innerHTML: claims are untrusted text.
 */

const BACKEND = "http://localhost:8000";
const MAX_CHARS = 5000;

const BAND = {
  trustworthy: ["var(--true)", "Likely trustworthy"],
  caution: ["var(--mixed)", "Treat with caution"],
  low: ["var(--false)", "Low trust"],
  insufficient_evidence: ["var(--none)", "Not enough evidence"],
};
const VERDICT = {
  false: "false", mixed: "mixed", true: "true",
  related: "related only", no_match: "no match",
};

const $ = (id) => document.getElementById(id);
const main = () => $("main");
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const factorName = (f) => f.replace(/_/g, " ");

/* ---------- DOM helpers (no innerHTML anywhere) ---------- */

function el(tag, opts = {}, ...children) {
  const node = document.createElement(tag);
  if (opts.class) node.className = opts.class;
  if (opts.text != null) node.textContent = opts.text;
  if (opts.style) node.setAttribute("style", opts.style);
  if (opts.href) { node.href = opts.href; node.target = "_blank"; node.rel = "noreferrer"; }
  for (const [k, v] of Object.entries(opts.attrs ?? {})) node.setAttribute(k, v);
  children.filter(Boolean).forEach((c) => node.append(c));
  return node;
}
const clear = () => main().replaceChildren();

/* ---------- states ---------- */

function showWorking(message) {
  clear();
  main().append(el("div", { class: "working" }, el("span", { class: "pulse" }),
                    document.createTextNode(message)));
}

function showError(message, offline) {
  clear();
  main().append(el("div", { class: "notice alert" },
    el("span", { class: "notice-title", text: offline ? "Analysis service offline" : "Couldn't check that" }),
    document.createTextNode(message)));
  if (offline) {
    main().append(el("p", { class: "small muted", style: "margin-top:10px" },
      document.createTextNode("Start it with "), el("code", { text: "./run.sh" }),
      document.createTextNode(" in the project folder, then reopen this popup.")));
  }
  $("again").hidden = false;
}

function showComposer(note) {
  clear();
  $("again").hidden = true;
  const box = el("textarea", {
    attrs: { placeholder: "Paste a sentence, a headline, or a whole paragraph…" },
  });
  const go = el("button", { class: "btn", text: "Check" });
  go.addEventListener("click", () => {
    const text = box.value.trim();
    if (text) analyze(text, null, "pasted text");
  });
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go.click();
  });
  main().append(
    el("p", { class: "small muted", style: "margin:0 0 10px",
              text: note ?? "Select text on any page and open this popup to check it "
                          + "automatically, or paste it below." }),
    box, go);
  box.focus();
}

/* ---------- the signature: evidence meter ---------- */

function evidenceMeter(explanation) {
  const segments = (explanation ?? []).filter((e) => e.weight > 0);
  const used = segments.reduce((a, e) => a + e.weight, 0);
  const unmeasured = Math.max(0, 1 - used);

  const bar = el("div", {
    class: "meter",
    attrs: {
      role: "img",
      "aria-label": `${Math.round(used * 100)} percent of possible evidence was measured`,
    },
  });
  segments.forEach((e) =>
    bar.append(el("div", { class: `meter-seg ${e.direction}`, style: `width:${e.weight * 100}%` })));
  if (unmeasured > 0.001) bar.append(el("div", { class: "meter-seg unmeasured" }));

  const key = el("div", { class: "meter-key" });
  segments.forEach((e) =>
    key.append(el("span", { class: "meter-key-item" },
      el("span", { class: `swatch ${e.direction}` }),
      document.createTextNode(`${factorName(e.factor)} `),
      el("b", { text: `${Math.round(e.weight * 100)}%` }))));
  if (unmeasured > 0.001) {
    key.append(el("span", { class: "meter-key-item" },
      el("span", { class: "swatch unmeasured" }),
      document.createTextNode("unmeasured "),
      el("b", { text: `${Math.round(unmeasured * 100)}%` })));
  }

  return el("div", {}, bar,
    el("p", { class: "meter-caption",
              text: "Bar width is how much evidence backed this score. Hatched space is "
                  + "evidence that was not available." }),
    key);
}


const LIVE_LABEL = {
  supported: ["Supported by a live source", "var(--true)"],
  contradicted: ["Contradicted by a live source", "var(--false)"],
  inconclusive: ["Live sources were unclear", "var(--none)"],
  no_evidence: ["No live source found", "var(--none)"],
  disabled: ["Live research is off", "var(--none)"],
};

/* Live evidence is drawn differently from a matched fact-check on purpose: it is a model
   reading an encyclopedia, not a professional's published verdict. */
function liveBlock(live) {
  const [label, color] = LIVE_LABEL[live.status] ?? [live.status, "var(--none)"];
  const decided = live.status === "supported" || live.status === "contradicted";
  const box = el("div", { class: "live" },
    el("span", { class: "live-head" },
      el("span", { class: "live-dot", style: `background:${color}` }),
      document.createTextNode(`Looked up live \u00b7 ${label}`)));
  if (decided) {
    const meta = el("div", { class: "evidence-meta" },
      el("a", { href: live.source_url, text: `Wikipedia: ${live.source_title}` }),
      el("span", { class: "num", text: `${Math.round((live.confidence ?? 0) * 100)}% confidence` }),
      el("span", { class: "num", text: `${Math.round((live.relevance ?? 0) * 100)}% relevance` }));
    box.append(el("p", { class: "evidence-claim", text: live.passage }), meta);
  }
  box.append(el("p", { class: "live-note", text: live.note ?? "" }));
  return box;
}

/* ---------- result ---------- */

function claimCard(c) {
  const card = el("div", { class: "claim" });

  card.append(el("div", { class: "claim-head" },
    el("p", { class: "claim-text", text: c.text }),
    el("span", { class: `tag ${c.verdict}`, text: VERDICT[c.verdict] ?? c.verdict })));

  if (c.matched_claim) {
    const meta = el("div", { class: "evidence-meta" });
    if (c.publisher) meta.append(el("span", { text: c.publisher }));
    if (c.rating) meta.append(el("span", { text: `rated ${c.rating}` }));
    if (c.similarity != null) {
      meta.append(el("span", { class: "num", text: `${Math.round(c.similarity * 100)}% similar` }));
    }
    if (c.source_url) meta.append(el("a", { href: c.source_url, text: "Open source" }));

    card.append(el("div", { class: "evidence" },
      el("span", { class: "eyebrow", text: "Matched fact-check" }),
      el("p", { class: "evidence-claim", text: c.matched_claim }),
      meta));
  }

  if (c.live) card.append(liveBlock(c.live));

  (c.warnings ?? []).forEach((w) =>
    card.append(el("div", { class: "guard" },
      el("span", { class: "eyebrow", text: "Verdict withheld" }),
      document.createTextNode(w))));

  // the guard box and the evidence meta already say it; only add prose when they don't
  if (c.explanation && !(c.warnings ?? []).length && !c.matched_claim) {
    card.append(el("p", { class: "claim-note", text: c.explanation }));
  }

  if ((c.signals ?? []).length) {
    card.append(el("details", {},
      el("summary", { text: "Why this sentence was checked" }),
      el("div", { class: "reveal-body" },
        el("div", { text: `Check-worthiness ${c.checkworthiness} — checked at 0.40 or above.` }),
        el("ul", {}, ...c.signals.map((s) => el("li", { text: s }))))));
  }
  return card;
}

function showResult(r, sourceLabel, sourceText) {
  clear();
  $("again").hidden = false;

  if (sourceText) {
    main().append(el("div", { class: "source" },
      el("span", { class: "eyebrow", text: `Checked ${sourceLabel}` }),
      el("div", { text: sourceText })));
  }
  if (r.truncated) {
    main().append(el("div", { class: "guard", style: "margin-bottom:14px" },
      el("span", { class: "eyebrow", text: "Shortened" }), document.createTextNode(r.truncated)));
  }

  const [color, label] = BAND[r.band] ?? ["var(--none)", r.band];
  const claims = r.fact_check?.claims ?? [];

  main().append(el("div", { class: "readout" },
    el("div", { class: "verdict-head" },
      el("div", { class: "score", style: `color:${color}` },
        document.createTextNode(String(r.trust_score)),
        el("span", { class: "score-max", text: " / 100" })),
      el("div", {},
        el("div", { class: "band", style: `color:${color}`, text: label }),
        el("div", { class: "verdict-note",
          text: `${claims.length} claim${claims.length === 1 ? "" : "s"} checked · `
              + `${r.fact_check?.claims_matched ?? 0} matched · ${r.timing_ms.total} ms` }))),
    evidenceMeter(r.explanation)));

  if (claims.length) {
    main().append(el("h2", { class: "eyebrow", text: `Claims found (${claims.length})` }));
    claims.forEach((c) => main().append(claimCard(c)));
  }

  if (r.explanation?.length) {
    const rows = el("div", {});
    r.explanation.forEach((e) =>
      rows.append(el("div", { class: "method-row" },
        el("div", {},
          el("div", { class: "method-factor", text: factorName(e.factor) }),
          el("span", { class: `dir ${e.direction}`, text: e.direction })),
        el("div", { class: "method-detail", text: e.detail }))));
    main().append(
      el("details", { style: "margin-top:16px" },
        el("summary", { text: "How this score was reached" }),
        el("div", { style: "margin-top:8px" }, rows)));
  }

  main().append(el("p", { class: "small muted", style: "margin-top:14px" },
    document.createTextNode(
      `${r.timing_ms.total} ms · ${r.timing_ms.speedup_vs_serial}× faster than running the `
      + "three models one after another · all on this machine")));
}

/* ---------- flow ---------- */

async function analyze(text, url, sourceLabel) {
  showWorking("Three models are reading this on your machine");
  const res = await chrome.runtime.sendMessage({ type: "TRUTHLENS_ANALYZE_TEXT", text, url });
  if (!res || res.error) return showError(res?.error ?? "Analysis failed.", res?.offline);
  showResult(res.report, sourceLabel, text.length > 260 ? `${text.slice(0, 260)}…` : text);
}

async function readSelection() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return { text: "", url: null };
  try {
    const [{ result } = {}] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
    return { text: (result || "").trim(), url: tab.url };
  } catch {
    return { text: "", url: tab.url, restricted: true };
  }
}

function pollHealth() {
  fetch(`${BACKEND}/health`)
    .then((r) => r.json())
    .then((h) => {
      $("dot").className = "dot up";
      $("health").textContent =
        `${Object.values(h.engines).filter((e) => e.status === "loaded").length}/3 models · `
        + `${h.corpus_claims.toLocaleString()} fact-checks`;
    })
    .catch(() => {
      $("dot").className = "dot down";
      $("health").textContent = "service offline";
    });
}

$("again").addEventListener("click", () => showComposer());

(async function start() {
  pollHealth();
  showWorking("Reading your selection");
  const { text, url, restricted } = await readSelection();
  if (text) {
    await analyze(text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text, url, "your selection");
  } else {
    showComposer(restricted
      ? "This browser page doesn't allow reading selections. Paste the text below instead."
      : undefined);
  }
})();
