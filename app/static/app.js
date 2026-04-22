const el = {
  artifact: document.getElementById("artifact"),
  artifactInfo: document.getElementById("artifact-info"),
  form: document.getElementById("score-form"),
  file: document.getElementById("file"),
  submit: document.getElementById("submit-btn"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  summary: document.getElementById("summary"),
  metrics: document.getElementById("metrics"),
  preview: document.getElementById("preview"),
  download: document.getElementById("download"),
};

let artifacts = [];

function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "number" && Math.abs(v) >= 1e6) return v.toExponential(2);
  return typeof v === "number" ? v.toFixed(digits) : v;
}

function setStatus(text, kind = "") {
  el.status.textContent = text;
  el.status.className = "status " + kind;
}

async function loadArtifacts() {
  setStatus("loading artifacts…");
  const res = await fetch("/artifacts");
  if (!res.ok) {
    setStatus("failed to load artifacts", "err");
    return;
  }
  const data = await res.json();
  artifacts = data.artifacts || [];
  el.artifact.innerHTML = "";
  if (!artifacts.length) {
    el.artifact.innerHTML = `<option value="">(no artifacts found — train one with experiments/run.py first)</option>`;
    el.artifactInfo.textContent = "Checkpoints directory empty.";
    el.submit.disabled = true;
    setStatus("");
    return;
  }
  for (const a of artifacts) {
    const opt = document.createElement("option");
    opt.value = a.id;
    opt.textContent = `${a.model_name} · ${a.trained_on} · seed ${a.seed} (${a.id})`;
    el.artifact.appendChild(opt);
  }
  el.submit.disabled = false;
  renderArtifactInfo();
  setStatus("");
}

function renderArtifactInfo() {
  const a = artifacts.find((x) => x.id === el.artifact.value);
  if (!a) {
    el.artifactInfo.textContent = "";
    return;
  }
  el.artifactInfo.innerHTML =
    `features: <code>${a.feature_count}</code> · ` +
    `window: <code>${a.window ?? "—"}</code> · ` +
    `threshold: <code>${fmt(a.threshold, 6)}</code> (${a.threshold_strategy}) · ` +
    `config: <code>${a.config_hash}</code> · ` +
    `git: <code>${a.git_sha.slice(0, 7)}</code>`;
}

el.artifact.addEventListener("change", renderArtifactInfo);

el.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!el.file.files[0]) {
    setStatus("pick a file first", "err");
    return;
  }
  el.submit.disabled = true;
  setStatus("scoring…");
  el.results.classList.add("hidden");

  const fd = new FormData();
  fd.append("artifact_id", el.artifact.value);
  fd.append("file", el.file.files[0]);

  let res;
  try {
    res = await fetch("/score", { method: "POST", body: fd });
  } catch (e) {
    setStatus("network error: " + e.message, "err");
    el.submit.disabled = false;
    return;
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail || body;
    let msg = `server error (${res.status}): `;
    if (typeof detail === "string") msg += detail;
    else if (detail.error === "schema_mismatch") {
      msg += `schema mismatch.\n  missing:    ${(detail.missing || []).join(", ") || "(none)"}\n  unexpected: ${(detail.unexpected || []).join(", ") || "(none)"}`;
    } else msg += JSON.stringify(detail);
    setStatus(msg, "err");
    el.submit.disabled = false;
    return;
  }

  const data = await res.json();
  renderResults(data);
  setStatus("done", "ok");
  el.submit.disabled = false;
});

function renderResults(data) {
  el.results.classList.remove("hidden");
  el.summary.innerHTML = `
    <ul class="muted" style="list-style: none; padding: 0; margin: 0;">
      <li>input rows: <code>${data.n_input_rows}</code> · scored: <code>${data.n_scored}</code> ${data.windowed ? "windows" : "rows"} · flagged: <code>${data.n_flagged}</code></li>
      <li>artifact threshold: <code>${fmt(data.threshold, 6)}</code></li>
    </ul>`;

  if (!data.metrics) {
    el.metrics.innerHTML = `<p class="muted">No labels in the uploaded file — skipping metrics.</p>`;
  } else {
    const rows = Object.entries(data.metrics).map(([family, m]) => {
      const f1 = family === "etapr" ? m.etapr_f1 : m.f1;
      const p = family === "etapr" ? m.tap : m.precision;
      const r = family === "etapr" ? m.tar : m.recall;
      return `<tr><td>${family}</td><td>${fmt(p)}</td><td>${fmt(r)}</td><td>${fmt(f1)}</td><td>${fmt(m.roc_auc)}</td><td>${fmt(m.pr_auc)}</td></tr>`;
    }).join("");
    el.metrics.innerHTML = `
      <table>
        <thead><tr><th>metric</th><th>P</th><th>R</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  const hasLabel = data.preview.length && data.preview[0].label !== null && data.preview[0].label !== undefined;
  const header = `<tr><th>row</th><th>score</th><th>flag</th>${hasLabel ? "<th>label</th>" : ""}</tr>`;
  const body = data.preview.map((r) => {
    const flagClass = r.flag ? ' class="flag"' : "";
    return `<tr${flagClass}><td>${r.index}</td><td>${fmt(r.score, 6)}</td><td>${r.flag}</td>${hasLabel ? `<td>${r.label ?? "—"}</td>` : ""}</tr>`;
  }).join("");
  el.preview.innerHTML = `<table><thead>${header}</thead><tbody>${body}</tbody></table>`;

  el.download.href = data.download_url;
}

loadArtifacts();
