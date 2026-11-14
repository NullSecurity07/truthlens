/* TruthLens in-page overlay.
 *
 * Rendered inside a shadow root so the host page's CSS — however aggressive —
 * cannot reach in, and ours cannot leak onto the page.
 *
 * Two things worth knowing:
 *  - @font-face is ignored inside a shadow root, so the two bundled faces are
 *    DECLARED in the host document under private family names. Declaring a face
 *    renders nothing by itself, so the page is unaffected, and the overlay gets
 *    the same typography as the popup and the dashboard instead of whatever
 *    fallback the visitor's machine happens to have.
 *  - createElement/textContent throughout, because the matched claims and the
 *    page's own selected text are untrusted input.
 */

(() => {
  if (window.__truthlensInjected) return;
  window.__truthlensInjected = true;

  const HOST_ID = "truthlens-overlay-host";
  const BAND = {
    trustworthy: ["#067647", "Likely trustworthy"],
    caution: ["#b54708", "Treat with caution"],
    low: ["#b42318", "Low trust"],
    insufficient_evidence: ["#667085", "Not enough evidence"],
  };
  const VERDICT = {
    false: "false", mixed: "mixed", true: "true",
    related: "related only", no_match: "no match",
  };
  const factorName = (f) => f.replace(/_/g, " ");

  /* ---------- DOM helpers ---------- */

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

  /* @font-face is ignored inside a shadow root (per spec), so the family has to be
     DECLARED in the host document. Declaring a face changes nothing on the page by
     itself — only an element that references the family is affected, and the name is
     private to us — so the overlay stays visually self-contained while still getting
     the same typography as the popup and dashboard. */
  function ensureFonts() {
    if (document.getElementById("truthlens-font-face")) return;
    let sans, serif;
    try {
      sans = chrome.runtime.getURL("fonts/public-sans.woff2");
      serif = chrome.runtime.getURL("fonts/instrument-serif.woff2");
    } catch {
      return;                       // not in an extension context; system stack is fine
    }
    const style = document.createElement("style");
    style.id = "truthlens-font-face";
    style.textContent =
      `@font-face{font-family:"TruthLens Sans";src:url("${sans}") format("woff2");` +
      `font-weight:300 800;font-display:swap}` +
      `@font-face{font-family:"TruthLens Serif";src:url("${serif}") format("woff2");` +
      `font-weight:400;font-display:swap}`;
    document.head?.appendChild(style);
  }

  function root() {
    ensureFonts();
    let host = document.getElementById(HOST_ID);
    if (host) return host.shadowRoot;
    host = document.createElement("div");
    host.id = HOST_ID;
    host.style.cssText = "all:initial;position:fixed;z-index:2147483647;bottom:22px;right:22px;";
    (document.body || document.documentElement).appendChild(host);
    const shadow = host.attachShadow({ mode: "open" });
    shadow.append(el("style", { text: CSS }), el("div", { class: "card" }));
    return shadow;
  }

  const close = () => document.getElementById(HOST_ID)?.remove();

  function paint(...body) {
    const shadow = root();
    const card = shadow.querySelector(".card");
    card.replaceChildren(
      el("header", {},
        el("span", { class: "wordmark" },
          el("span", { class: "lens-mark" }), document.createTextNode("TruthLens")),
        (() => {
          const x = el("button", { class: "x", text: "×", attrs: { "aria-label": "Close" } });
          x.addEventListener("click", close);
          return x;
        })()),
      el("div", { class: "body" }, ...body));
    return card;
  }

  /* ---------- states ---------- */

  const working = (msg) =>
    paint(el("div", { class: "working" }, el("span", { class: "pulse" }), document.createTextNode(msg)));

  function failed(message, offline) {
    const parts = [el("div", { class: "notice" },
      el("span", { class: "notice-title", text: offline ? "Analysis service offline" : "Couldn't check that" }),
      document.createTextNode(message))];
    if (offline) {
      parts.push(el("p", { class: "hint" },
        document.createTextNode("Start it with "), el("code", { text: "./run.sh" }),
        document.createTextNode(" in the project folder, then try again.")));
    }
    paint(...parts);
  }

  /* ---------- the signature: evidence meter ---------- */

  function evidenceMeter(explanation) {
    const segs = (explanation ?? []).filter((e) => e.weight > 0);
    const used = segs.reduce((a, e) => a + e.weight, 0);
    const unmeasured = Math.max(0, 1 - used);

    const bar = el("div", { class: "meter", attrs: { role: "img",
      "aria-label": `${Math.round(used * 100)} percent of possible evidence was measured` } });
    segs.forEach((e) =>
      bar.append(el("div", { class: `seg ${e.direction}`, style: `width:${e.weight * 100}%` })));
    if (unmeasured > 0.001) bar.append(el("div", { class: "seg unmeasured" }));

    const key = el("div", { class: "key" });
    segs.forEach((e) =>
      key.append(el("span", { class: "key-item" }, el("span", { class: `swatch ${e.direction}` }),
        document.createTextNode(`${factorName(e.factor)} `),
        el("b", { text: `${Math.round(e.weight * 100)}%` }))));
    if (unmeasured > 0.001) {
      key.append(el("span", { class: "key-item" }, el("span", { class: "swatch unmeasured" }),
        document.createTextNode("unmeasured "), el("b", { text: `${Math.round(unmeasured * 100)}%` })));
    }
    return el("div", {}, bar,
      el("p", { class: "meter-caption",
        text: "Bar width is how much evidence backed this score. Hatched space is evidence "
            + "that was not available." }), key);
  }

  /* ---------- live evidence ---------- */
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

  /* ---------- claim record ---------- */

  function claimCard(c) {
    const card = el("div", { class: "claim" },
      el("div", { class: "claim-head" },
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
        el("p", { class: "evidence-claim", text: c.matched_claim }), meta));
    }

    if (c.live) card.append(liveBlock(c.live));

    (c.warnings ?? []).forEach((w) =>
      card.append(el("div", { class: "guard" },
        el("span", { class: "eyebrow", text: "Verdict withheld" }), document.createTextNode(w))));

    if (c.explanation && !(c.warnings ?? []).length && !c.matched_claim) {
      card.append(el("p", { class: "claim-note", text: c.explanation }));
    }
    return card;
  }

  /* ---------- result ---------- */

  function result(r) {
    const [color, label] = BAND[r.band] ?? ["#667085", r.band];
    const claims = r.fact_check?.claims ?? [];
    const blocks = [];

    if (r.truncated) {
      blocks.push(el("div", { class: "guard" },
        el("span", { class: "eyebrow", text: "Shortened" }), document.createTextNode(r.truncated)));
    }

    blocks.push(
      el("div", { class: "verdict-head" },
        el("div", { class: "score", style: `color:${color}` },
          document.createTextNode(String(r.trust_score)),
          el("span", { class: "score-max", text: " / 100" })),
        el("div", {},
          el("div", { class: "band", style: `color:${color}`, text: label }),
          el("div", { class: "verdict-note",
            text: `${claims.length} claim${claims.length === 1 ? "" : "s"} checked · `
                + `${r.fact_check?.claims_matched ?? 0} matched` }))),
      evidenceMeter(r.explanation));

    if (claims.length) {
      blocks.push(el("h2", { class: "eyebrow section", text: `Claims found (${claims.length})` }));
      claims.forEach((c) => blocks.push(claimCard(c)));
    }

    if (r.explanation?.length) {
      const rows = el("div", { class: "reveal-body" });
      r.explanation.forEach((e) =>
        rows.append(el("div", { class: "method-row" },
          el("span", { class: "method-factor", text: factorName(e.factor) }),
          el("span", { class: `dir ${e.direction}`, text: e.direction }),
          el("span", { class: "method-detail", text: e.detail }))));
      blocks.push(el("details", {},
        el("summary", { text: "How this score was reached" }), rows));
    }

    blocks.push(el("p", { class: "foot",
      text: `${r.timing_ms.total} ms · ${r.timing_ms.speedup_vs_serial}× faster than running the `
          + "three models in sequence · all on this machine" }));

    paint(...blocks);
  }

  /* ---------- styles ---------- */

  const CSS = `
  :host, * { box-sizing: border-box; }
  .card {
    --paper:#fcfcfd; --card:#fff; --sunken:#f6f7f9; --rule:#e4e7ec; --rule-soft:#eef0f3;
    --ink:#10151f; --graphite:#5a6472; --mist:#667080;
    --lens:#0e6b70; --false:#b42318; --mixed:#b54708; --true:#067647;
    --related:#5925dc; --none:#667085;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
    width: 390px; max-height: 80vh; overflow-y: auto; overflow-x: hidden;
    background: var(--paper); color: var(--ink);
    font: 400 13px/1.55 "TruthLens Sans", ui-sans-serif, system-ui, -apple-system,
          "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-variant-numeric: tabular-nums;
    border: 1px solid var(--rule); border-radius: 12px;
    box-shadow: 0 1px 2px rgba(16,21,31,.04), 0 16px 44px rgba(16,21,31,.14);
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 15px; border-bottom: 1px solid var(--rule);
    position: sticky; top: 0; background: var(--paper); z-index: 2;
  }
  .wordmark { display: flex; align-items: center; gap: 7px; font-weight: 600;
              font-size: 13.5px; letter-spacing: -.01em; }
  .lens-mark { width: 10px; height: 10px; border-radius: 50%;
               border: 2.5px solid var(--lens); flex: none; }
  .x { all: unset; cursor: pointer; font-size: 19px; line-height: 1; color: var(--mist);
       padding: 0 3px; }
  .x:hover { color: var(--ink); }
  .body { padding: 15px; }
  .eyebrow { font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em;
             text-transform: uppercase; color: var(--mist); display: block; margin-bottom: 4px; }
  .eyebrow.section { margin: 17px 0 9px; }
  .hint { color: var(--graphite); font-size: 12px; margin: 9px 0 0; }
  code { font-family: var(--mono); font-size: 11px; background: var(--sunken);
         border: 1px solid var(--rule); border-radius: 4px; padding: 1px 5px; }
  a { color: var(--lens); text-underline-offset: 2px; }

  .verdict-head { display: flex; align-items: baseline; gap: 13px; margin-bottom: 13px; }
  .score { font: 400 46px/.78 "TruthLens Serif", ui-serif, Georgia, "Times New Roman", serif;
           letter-spacing: -.02em; }
  .score-max { font: 400 12px "TruthLens Sans", ui-sans-serif, system-ui, sans-serif;
               color: var(--mist); }
  .band { font-size: 14.5px; font-weight: 700; letter-spacing: -.01em; }
  .verdict-note { color: var(--graphite); font-size: 11.5px; }

  .meter { display: flex; height: 17px; border-radius: 4px; overflow: hidden;
           background: var(--sunken); border: 1px solid var(--rule); }
  .seg.lowers { background: var(--false); }
  .seg.raises { background: var(--true); }
  .seg.neutral { background: var(--none); }
  .seg.unmeasured { flex: 1;
    background-image: repeating-linear-gradient(-45deg, #fff 0 5px, #d7dce4 5px 7px); }
  .meter-caption { margin: 8px 0 0; font-size: 10.5px; color: var(--mist); line-height: 1.45; }
  .key { display: flex; flex-wrap: wrap; gap: 5px 13px; margin-top: 8px;
         font-size: 11px; color: var(--graphite); }
  .key-item { display: flex; align-items: center; gap: 5px; }
  .swatch { width: 9px; height: 9px; border-radius: 2px; flex: none; }
  .swatch.lowers { background: var(--false); }
  .swatch.raises { background: var(--true); }
  .swatch.neutral { background: var(--none); }
  .swatch.unmeasured { border: 1px solid var(--rule);
    background-image: repeating-linear-gradient(-45deg, #fff 0 3px, #d7dce4 3px 5px); }
  .key b { font-family: var(--mono); font-size: 10px; font-weight: 600; color: var(--ink); }


  .live { margin-top: 10px; padding: 9px 11px; border: 1px dashed var(--rule);
          border-radius: 8px; background: var(--sunken); }
  .live-head { display: flex; align-items: center; gap: 6px; font-family: var(--mono);
               font-size: 9.5px; letter-spacing: .1em; text-transform: uppercase;
               color: var(--graphite); margin-bottom: 6px; }
  .live-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
  .live-note { margin: 6px 0 0; font-size: 11px; color: var(--mist); line-height: 1.45; }
  .claim { background: var(--card); border: 1px solid var(--rule); border-radius: 9px;
           padding: 11px 12px; margin-bottom: 8px; }
  .claim-head { display: flex; gap: 9px; align-items: flex-start; }
  .claim-text { flex: 1; margin: 0; font-size: 12.5px; line-height: 1.45; }
  .tag { flex: none; display: inline-flex; align-items: center; gap: 4px;
         font-family: var(--mono); font-size: 8.5px; font-weight: 600; letter-spacing: .07em;
         text-transform: uppercase; padding: 3px 7px; border-radius: 4px;
         border: 1px solid transparent; white-space: nowrap; }
  .tag::before { content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
  .tag.false   { color: var(--false);   background: #fef3f2; border-color: #fecdca; }
  .tag.mixed   { color: var(--mixed);   background: #fffaeb; border-color: #fedf89; }
  .tag.true    { color: var(--true);    background: #ecfdf3; border-color: #abefc6; }
  .tag.related { color: var(--related); background: #f4f3ff; border-color: #d9d6fe; }
  .tag.no_match{ color: var(--none);    background: #f2f4f7; border-color: var(--rule); }

  .evidence { margin-top: 10px; padding-left: 11px; border-left: 2px solid var(--rule); }
  .evidence-claim { margin: 0; font-size: 12px; line-height: 1.45; }
  .evidence-meta { margin-top: 5px; font-size: 10.5px; color: var(--graphite);
                   display: flex; flex-wrap: wrap; gap: 3px 9px; align-items: center; }
  .evidence-meta .num { font-family: var(--mono); }

  .guard { margin-top: 9px; padding: 8px 10px; border-radius: 7px; background: #fffaeb;
           border: 1px solid #fedf89; font-size: 11.5px; line-height: 1.45; color: #93370d; }
  .guard .eyebrow { color: #93370d; margin-bottom: 2px; }
  .claim-note { margin: 9px 0 0; font-size: 11.5px; color: var(--graphite); }

  details { margin-top: 14px; }
  summary { cursor: pointer; font-size: 11.5px; color: var(--lens); list-style: none;
            display: inline-flex; align-items: center; gap: 4px; }
  summary::-webkit-details-marker { display: none; }
  summary::before { content: "+"; font-family: var(--mono); }
  details[open] summary::before { content: "\\2212"; }
  .reveal-body { margin-top: 8px; }
  .method-row { display: grid; grid-template-columns: 74px 52px 1fr; gap: 8px;
                padding: 7px 0; border-bottom: 1px solid var(--rule-soft); font-size: 11px; }
  .method-row:last-child { border-bottom: 0; }
  .method-factor { font-weight: 600; }
  .dir { font-family: var(--mono); font-size: 8.5px; letter-spacing: .05em; text-transform: uppercase; }
  .dir.lowers { color: var(--false); } .dir.raises { color: var(--true); } .dir.neutral { color: var(--mist); }
  .method-detail { color: var(--graphite); line-height: 1.45; }

  .notice { padding: 11px 12px; border-radius: 9px; border: 1px solid #fecdca;
            background: #fef3f2; color: #912018; font-size: 12.5px; }
  .notice-title { display: block; font-weight: 700; margin-bottom: 2px; }

  .working { display: flex; align-items: center; justify-content: center; gap: 9px;
             color: var(--graphite); font-size: 12.5px; padding: 20px 0; }
  .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--lens); flex: none;
           animation: pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: .25; transform: scale(.8); } 50% { opacity: 1; transform: scale(1); } }
  @media (prefers-reduced-motion: reduce) { .pulse { animation: none; opacity: .7; } }

  .foot { margin: 14px 0 0; padding-top: 10px; border-top: 1px solid var(--rule-soft);
          font-size: 10.5px; color: var(--mist); line-height: 1.5; }
  `;

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type === "TRUTHLENS_LOADING") {
      working("Three models are reading this on your machine");
    } else if (msg?.type === "TRUTHLENS_RESULT") {
      if (msg.error) failed(msg.error, msg.offline);
      else result(msg.report);
    }
  });
})();
