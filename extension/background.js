/* TruthLens service worker: context menu -> backend -> overlay in the page. */

const BACKEND = "http://localhost:8000";
const MAX_CHARS = 5000;
const MENU_ID = "truthlens-analyze";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: 'Analyze "%s" with TruthLens',
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === MENU_ID && tab?.id != null) {
    analyze(tab.id, info.selectionText || "", tab.url);
  }
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "analyze-selection") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  const [{ result } = {}] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection().toString(),
  });
  analyze(tab.id, result || "", tab.url);
});

/* The popup analyses text the user pasted, with no tab involved. */
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "TRUTHLENS_ANALYZE_TEXT") {
    callBackend(msg.text, msg.url).then(sendResponse);
    return true; // keep the channel open for the async reply
  }
});

async function callBackend(text, url) {
  const trimmed = (text || "").trim();
  if (!trimmed) return { error: "Nothing was selected." };

  const truncated = trimmed.length > MAX_CHARS;
  const payload = truncated ? trimmed.slice(0, MAX_CHARS) : trimmed;

  try {
    const resp = await fetch(`${BACKEND}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: payload, url: url || null }),
    });
    if (!resp.ok) {
      let detail = `Backend returned HTTP ${resp.status}.`;
      try {
        const body = await resp.json();
        if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
        else if (typeof body.detail === "string") detail = body.detail;
        else if (body.detail?.message) detail = body.detail.message;
      } catch (_) { /* body was not JSON; keep the status message */ }
      return { error: detail };
    }
    const report = await resp.json();
    if (truncated) {
      report.truncated = `Selection was ${trimmed.length} characters; only the first ${MAX_CHARS} were analyzed.`;
    }
    return { report };
  } catch (_err) {
    return {
      error: "TruthLens backend offline - start it with ./run.sh in the project folder.",
      offline: true,
    };
  }
}

async function analyze(tabId, text, url) {
  await send(tabId, { type: "TRUTHLENS_LOADING" });
  const result = await callBackend(text, url);
  await send(tabId, { type: "TRUTHLENS_RESULT", ...result });
}

/* Content scripts are absent on restricted pages (chrome://, the web store, PDFs).
   Inject on demand, and fall back to a notification if even that is refused. */
async function send(tabId, message) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch (_) {
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
      await chrome.tabs.sendMessage(tabId, message);
    } catch (_again) {
      if (message.type === "TRUTHLENS_RESULT") {
        chrome.notifications?.create({
          type: "basic",
          iconUrl: "icon128.png",
          title: "TruthLens",
          message: message.error
            ? message.error
            : `Trust Score ${message.report.trust_score}/100 (${message.report.band}). This page blocks overlays; open the dashboard to see details.`,
        });
      }
    }
  }
}
