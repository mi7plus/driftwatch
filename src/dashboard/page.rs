//! The self-contained dashboard web page.
//!
//! One HTML document with inlined CSS and vanilla JS — no external assets, no
//! build step. The script polls `api/report` (relative, so it works whether the
//! router is served at `/` or nested under a prefix) every couple of seconds and
//! re-renders in place.

/// Render the dashboard page with `title` substituted in.
pub(super) fn render(title: &str) -> String {
    PAGE.replace("{{TITLE}}", &escape(title))
}

/// Minimal HTML-escaping for the developer-supplied title.
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}} — drift dashboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #181b22; --border: #272b34; --fg: #e6e9ef;
    --muted: #9aa3b2; --ok: #2ea043; --ok-bg: #0f2a17; --bad: #f85149;
    --bad-bg: #2a1113; --accent: #58a6ff;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f6f8fa; --panel: #ffffff; --border: #d0d7de; --fg: #1f2328;
      --muted: #656d76; --ok: #1a7f37; --ok-bg: #dafbe1; --bad: #cf222e;
      --bad-bg: #ffebe9; --accent: #0969da;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 24px 16px 64px; }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 20px; margin: 0; }
  .meta { color: var(--muted); font-size: 13px; }
  .banner {
    margin: 18px 0; padding: 16px 20px; border-radius: 10px; font-weight: 600;
    font-size: 18px; border: 1px solid var(--border); display: flex;
    align-items: center; gap: 12px;
  }
  .banner.ok { background: var(--ok-bg); border-color: var(--ok); }
  .banner.bad { background: var(--bad-bg); border-color: var(--bad); }
  .banner.wait { color: var(--muted); }
  .dot { width: 12px; height: 12px; border-radius: 50%; flex: none; }
  .dot.ok { background: var(--ok); } .dot.bad { background: var(--bad); }
  .dot.wait { background: var(--muted); }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  td.num { font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .pill.ok { color: var(--ok); background: var(--ok-bg); }
  .pill.bad { color: var(--bad); background: var(--bad-bg); }
  .kind { color: var(--muted); font-size: 12px; }
  .metrics { color: var(--muted); font-size: 12px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 4px 4px 8px; margin-top: 16px; }
  .spark { margin-top: 16px; }
  .spark svg { width: 100%; height: 48px; display: block; }
  footer { margin-top: 28px; color: var(--muted); font-size: 12px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title">{{TITLE}}</h1>
    <div class="meta"><span id="meta">connecting…</span></div>
  </header>

  <div id="banner" class="banner wait"><span class="dot wait"></span><span id="banner-text">Waiting for the first drift check…</span></div>

  <div class="spark" id="spark"></div>

  <div class="card">
    <table>
      <thead>
        <tr><th>Feature</th><th>Verdict</th><th>Primary</th><th>Score</th><th>Threshold</th><th>Metrics</th></tr>
      </thead>
      <tbody id="rows">
        <tr><td colspan="6" class="kind">No data yet.</td></tr>
      </tbody>
    </table>
  </div>

  <footer>Auto-refreshing every 2s · <a href="api/report">raw JSON</a> · powered by driftwatch</footer>
</div>

<script>
const fmt = (x, d = 4) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));

function ago(secs) {
  if (!secs) return "never";
  const d = Math.max(0, Math.floor(Date.now() / 1000 - secs));
  if (d < 2) return "just now";
  if (d < 60) return d + "s ago";
  if (d < 3600) return Math.floor(d / 60) + "m ago";
  return Math.floor(d / 3600) + "h ago";
}

function sparkline(history) {
  if (!history || history.length < 2) return "";
  const w = 900, h = 48, n = history.length;
  const pts = history.map((p, i) => {
    const x = (i / (n - 1)) * w;
    const y = h - Math.max(0, Math.min(1, p.fraction)) * (h - 6) - 3;
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
  const dots = history.map((p, i) => {
    const x = (i / (n - 1)) * w;
    const y = h - Math.max(0, Math.min(1, p.fraction)) * (h - 6) - 3;
    const c = p.drifted ? "var(--bad)" : "var(--ok)";
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.2" fill="${c}"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline fill="none" stroke="var(--accent)" stroke-width="1.5" points="${pts}"/>${dots}</svg>
    <div class="meta">drifted-feature fraction over the last ${n} checks</div>`;
}

function render(d) {
  document.getElementById("title").textContent = d.title || "driftwatch";
  document.title = (d.title || "driftwatch") + " — drift dashboard";

  const banner = document.getElementById("banner");
  const bt = document.getElementById("banner-text");
  const dot = banner.querySelector(".dot");
  if (!d.has_data) {
    banner.className = "banner wait"; dot.className = "dot wait";
    bt.textContent = "Waiting for the first drift check…";
  } else if (d.dataset_drifted) {
    banner.className = "banner bad"; dot.className = "dot bad";
    bt.textContent = `Dataset drift detected — ${(d.drifted_fraction * 100).toFixed(0)}% of features drifted (threshold ${(d.dataset_fraction_threshold * 100).toFixed(0)}%)`;
  } else {
    banner.className = "banner ok"; dot.className = "dot ok";
    bt.textContent = `No dataset drift — ${(d.drifted_fraction * 100).toFixed(0)}% of features drifted (threshold ${(d.dataset_fraction_threshold * 100).toFixed(0)}%)`;
  }

  document.getElementById("meta").textContent =
    `${d.checks} check${d.checks === 1 ? "" : "s"} · updated ${ago(d.updated_secs)}`;

  document.getElementById("spark").innerHTML = sparkline(d.history);

  const rows = document.getElementById("rows");
  if (!d.features || d.features.length === 0) {
    rows.innerHTML = `<tr><td colspan="6" class="kind">No features.</td></tr>`;
    return;
  }
  rows.innerHTML = d.features.map(f => {
    const pill = f.verdict === "drifted"
      ? `<span class="pill bad">DRIFTED</span>`
      : `<span class="pill ok">stable</span>`;
    const metrics = f.metrics.map(m =>
      m.p_value === null || m.p_value === undefined
        ? `${m.metric} ${fmt(m.statistic)}`
        : `${m.metric} ${fmt(m.statistic)} (p=${fmt(m.p_value)})`
    ).join(" · ");
    return `<tr>
      <td>${escapeHtml(f.feature)} <span class="kind">${f.kind}</span></td>
      <td>${pill}</td>
      <td>${f.primary_metric}</td>
      <td class="num">${fmt(f.primary_score)}</td>
      <td class="num">${fmt(f.threshold, 3)}</td>
      <td class="metrics">${metrics}</td>
    </tr>`;
  }).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function refresh() {
  try {
    const r = await fetch("api/report", { cache: "no-store" });
    if (r.ok) render(await r.json());
  } catch (e) { /* transient; try again on the next tick */ }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"#;
