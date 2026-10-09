/*
 * result.html: renders the saved scan (verdict, confidence, Grad-CAM viewer,
 * model evaluation figures, photo quality) and the report actions.
 */

const OUTCOMES = {
  black_sigatoka: {
    kicker: "Symptoms detected",
    title: "Black Sigatoka likely",
    summary: "The leaf shows a pattern the model associates with Black Sigatoka (Pseudocercospora fijiensis).",
    meterLabel: "Model confidence",
    steps: [
      "Report it. Black Sigatoka is a notifiable plant pest in Australia. Call the Exotic Plant Pest Hotline on 1800 084 881.",
      "Don't move leaves, suckers or fruit off the property, and don't take samples yourself unless an officer asks you to.",
      "Photograph the whole plant and note its location so officers can find it.",
      "Wash hands, tools and boots before visiting other banana plants.",
    ],
  },
  healthy: {
    kicker: "No symptoms found",
    title: "Leaf looks healthy",
    summary: "The model found no pattern it associates with Black Sigatoka in this photo.",
    meterLabel: "Model confidence",
    steps: [
      "This covers the photographed leaf only. Check older, lower leaves too — symptoms usually appear there first.",
      "Early streaks are faint and easy to miss. Scan again if you see new specks or streaks over the next weeks.",
      "If you are worried about a plant regardless of this result, call the Exotic Plant Pest Hotline on 1800 084 881.",
    ],
  },
  not_black_sigatoka: {
    kicker: "Outside what the model knows",
    title: "Not recognised as Black Sigatoka",
    summary: "This photo doesn't closely match the healthy or Black Sigatoka banana leaves the model was trained on. It may be a different banana disease, a different plant, or a photo that isn't a leaf. This tool can't say which.",
    meterLabel: "Similarity to known banana leaves",
    steps: [
      "If this is a banana leaf with unusual symptoms, it could still be a serious problem. Get it looked at by an agronomist or call 1800 084 881.",
      "Retake the photo with a single leaf filling most of the frame, in even daylight.",
      "This tool doesn't identify other diseases by name.",
    ],
  },
};

const INFERNO = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];

document.addEventListener("DOMContentLoaded", () => {
  let scan;
  try { scan = JSON.parse(sessionStorage.getItem("lastScan")); } catch (_) { scan = null; }
  if (!scan || !scan.result) { window.location.replace("upload.html"); return; }

  const { result } = scan;
  const outcome = OUTCOMES[result.label];
  const $ = (id) => document.getElementById(id);

  /* ---------- Header ---------- */
  const when = new Date(scan.analysedAt);
  $("report-meta").textContent =
    `${scan.fileName}, analysed ${when.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}` +
    (result.id ? `, reference #${result.id}` : "") + (result.mocked ? " (demo result, not from the model)" : "");

  /* ---------- Verdict ---------- */
  const verdict = $("verdict");
  verdict.dataset.label = result.label;
  $("verdict-kicker").textContent = outcome.kicker;
  $("verdict-title").textContent = outcome.title;
  $("verdict-summary").textContent = outcome.summary;
  document.title = `${outcome.title} · Black Sigatoka Detector`;

  const pct = result.confidence * 100;
  $("meter-label").textContent = outcome.meterLabel;
  $("meter-value").textContent = `${pct.toFixed(1)}%`;
  const track = $("meter-track");
  track.setAttribute("aria-valuenow", pct.toFixed(1));
  track.setAttribute("aria-valuetext", `${pct.toFixed(1)} percent`);
  requestAnimationFrame(() => requestAnimationFrame(() => { $("meter-fill").style.width = `${pct}%`; }));

  const meterNote = $("meter-note");
  if (result.label === "not_black_sigatoka") {
    const needed = (1 - MODEL_INFO.gate.threshold) * 100;
    const marker = document.createElement("div");
    marker.className = "meter-marker";
    marker.style.left = `${needed}%`;
    marker.innerHTML = `<span>${needed.toFixed(0)}% needed</span>`;
    track.appendChild(marker);
    meterNote.textContent = `Photos must be at least ${needed.toFixed(0)}% similar to our training leaves before the classifier will judge them. This one wasn't, so no diagnosis was made and there is no heatmap.`;
    $("probs").hidden = true;
  } else {
    const band = pct >= 90 ? ["High", "The model is confident."]
      : pct >= 70 ? ["Moderate", "Consider retaking the photo or checking more leaves."]
      : ["Low", "Treat this as inconclusive. Retake the photo in better conditions."];
    meterNote.innerHTML = `<span class="band-chip"></span> `;
    meterNote.querySelector(".band-chip").textContent = `${band[0]} confidence`;
    meterNote.append(band[1]);

    // Two-class softmax: the other class gets the remainder
    const pBS = result.label === "black_sigatoka" ? result.confidence : 1 - result.confidence;
    setProb("prob-bs", pBS);
    setProb("prob-healthy", 1 - pBS);
  }

  function setProb(id, p) {
    const row = $(id);
    requestAnimationFrame(() => requestAnimationFrame(() => { row.querySelector("i").style.width = `${p * 100}%`; }));
    row.querySelector(".prob-val").textContent = `${(p * 100).toFixed(1)}%`;
  }

  const stepsList = $("next-steps-list");
  outcome.steps.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    stepsList.appendChild(li);
  });

  /* ---------- Photo + Grad-CAM viewer ---------- */
  const stage = $("viewer-stage");
  stage.dataset.fit = MODEL_INFO.heatmapFit;
  const photo = $("viewer-photo");
  if (scan.image) photo.src = scan.image;
  else photo.alt = "Photo not available";

  const modeButtons = [...document.querySelectorAll("[data-mode-btn]")];
  const opacity = $("opacity");
  const opacityOut = $("opacity-out");
  const canvas = $("heatmap-canvas");
  const probe = $("viewer-probe");
  let camValues = null; // Float32Array 0..1, heatmap resolution
  let camSize = [0, 0];

  function setMode(mode) {
    stage.dataset.mode = mode;
    modeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.modeBtn === mode)));
    $("opacity-row").hidden = mode !== "overlay";
    paintHeatmap();
  }
  modeButtons.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.modeBtn)));
  opacity.addEventListener("input", () => {
    opacityOut.textContent = `${opacity.value}%`;
    paintHeatmap();
  });

  if (result.heatmap_url) {
    loadHeatmap(result.heatmap_url).then(() => setMode("overlay")).catch(noHeatmap);
  } else {
    noHeatmap();
  }

  function noHeatmap() {
    modeButtons.forEach((b) => { if (b.dataset.modeBtn !== "photo") b.disabled = true; });
    setMode("photo");
    $("opacity-row").hidden = true;
    $("legend").hidden = true;
    $("heatmap-stats").hidden = true;
    document.querySelector(".viewer-controls").hidden = true;
    const msg = $("viewer-empty");
    msg.hidden = false;
    msg.textContent = result.label === "not_black_sigatoka"
      ? "No attention map: the classifier didn't run because the photo was outside what the model knows."
      : "The attention map couldn't be generated for this photo. The diagnosis above is unaffected.";
  }

  async function loadHeatmap(url) {
    // Fetch as a blob so the canvas isn't "tainted" by a cross-origin image.
    const blob = await (await fetch(url)).blob();
    const bmp = await createImageBitmap(blob);
    const w = bmp.width, h = bmp.height;
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(bmp, 0, 0);
    const { data } = tctx.getImageData(0, 0, w, h);
    camValues = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) camValues[i] = data[i * 4] / 255;
    camSize = [w, h];
    canvas.width = w; canvas.height = h;

    // Summary statistics
    let peak = 0, peakIdx = 0, focus = 0;
    for (let i = 0; i < camValues.length; i++) {
      if (camValues[i] > peak) { peak = camValues[i]; peakIdx = i; }
      if (camValues[i] >= 0.5) focus++;
    }
    const px = (peakIdx % w) / w, py = Math.floor(peakIdx / w) / h;
    const vert = py < 0.33 ? "top" : py > 0.66 ? "bottom" : "middle";
    const horiz = px < 0.33 ? "left" : px > 0.66 ? "right" : "centre";
    $("stat-focus").textContent = `${((focus / camValues.length) * 100).toFixed(0)}% of the frame`;
    $("stat-peak").textContent = vert === "middle" && horiz === "centre" ? "Centre" : `${vert[0].toUpperCase()}${vert.slice(1)} ${horiz}`;
  }

  function paintHeatmap() {
    if (!camValues) return;
    const ctx = canvas.getContext("2d");
    const [w, h] = camSize;
    const img = ctx.createImageData(w, h);
    const overlay = stage.dataset.mode === "overlay";
    const alphaScale = overlay ? Number(opacity.value) / 100 : 1;
    for (let i = 0; i < camValues.length; i++) {
      const v = camValues[i];
      const [r, g, b] = colormap(v);
      const o = i * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b;
      // In overlay mode, low-attention areas stay see-through so the leaf is visible
      img.data[o + 3] = overlay ? Math.round(255 * alphaScale * Math.min(1, Math.max(0, (v - 0.12) / 0.5))) : 235;
    }
    ctx.putImageData(img, 0, 0);
  }

  function colormap(v) {
    const t = Math.max(0, Math.min(0.9999, v)) * (INFERNO.length - 1);
    const i = Math.floor(t), f = t - i;
    const a = INFERNO[i], b = INFERNO[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  // Hover / touch readout of attention at a point
  function probeAt(clientX, clientY) {
    if (!camValues || stage.dataset.mode === "photo") { probe.style.display = "none"; return; }
    const rect = stage.getBoundingClientRect();
    const x = (clientX - rect.left) / rect.width, y = (clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) { probe.style.display = "none"; return; }
    const [w, h] = camSize;
    const v = camValues[Math.min(h - 1, Math.floor(y * h)) * w + Math.min(w - 1, Math.floor(x * w))];
    probe.textContent = `Attention ${(v * 100).toFixed(0)}%`;
    probe.style.left = `${x * 100}%`;
    probe.style.top = `${y * 100}%`;
    probe.style.display = "block";
  }
  stage.addEventListener("pointermove", (e) => probeAt(e.clientX, e.clientY));
  stage.addEventListener("pointerdown", (e) => probeAt(e.clientX, e.clientY));
  stage.addEventListener("pointerleave", () => { probe.style.display = "none"; });

  /* ---------- Model evaluation ---------- */
  const m = modelMetrics();
  $("sample-flag").hidden = !MODEL_INFO.isPlaceholder;
  $("model-name").textContent = MODEL_INFO.activeName;
  $("model-eval").textContent = `${MODEL_INFO.evaluatedOn}: ${m.total} photos (${m.positives} Black Sigatoka, ${m.negatives} healthy).`;

  const metricDefs = [
    ["Accuracy", m.accuracy, `Of all ${m.total} test photos, the share classified correctly.`],
    ["Recall", m.recall, `Of ${m.positives} infected leaves, the share the model caught. The key figure for biosecurity.`],
    ["Precision", m.precision, "When the model says Black Sigatoka, how often it's right."],
    ["F1 score", m.f1, "Balance of precision and recall in one number."],
    ["Specificity", m.specificity, `Of ${m.negatives} healthy leaves, the share correctly cleared.`],
  ];
  const metricsEl = $("metrics");
  metricDefs.forEach(([name, value, desc]) => {
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `<div class="metric-value"></div><div class="metric-name"></div><p class="metric-desc"></p>`;
    div.querySelector(".metric-value").textContent = formatPct(value);
    div.querySelector(".metric-name").textContent = name;
    div.querySelector(".metric-desc").textContent = desc;
    metricsEl.appendChild(div);
  });

  const cell = (id, n, of, note) => {
    const td = $(id);
    td.innerHTML = `${n}<small></small>`;
    td.querySelector("small").textContent = `${formatPct(of ? n / of : 0, 0)} ${note}`;
  };
  cell("cm-tp", m.tp, m.positives, "caught");
  cell("cm-fn", m.fn, m.positives, "missed");
  cell("cm-fp", m.fp, m.negatives, "false alarm");
  cell("cm-tn", m.tn, m.negatives, "cleared");

  $("explain-recall").textContent =
    `In testing, the model missed ${m.fn} of ${m.positives} infected leaves (${formatPct(m.fn / (m.positives || 1))}). ` +
    "A healthy result is therefore a strong signal, not a guarantee — which is why reporting any suspicion still matters.";
  $("explain-fp").textContent =
    `${m.fp} of ${m.negatives} healthy test leaves were wrongly flagged (${formatPct(m.fp / (m.negatives || 1))}). ` +
    "A positive result should be confirmed by a plant pathologist before any action beyond reporting.";

  /* ---------- Photo quality recap ---------- */
  const qList = $("quality-recap");
  (scan.quality || []).forEach((c) => {
    const li = document.createElement("li");
    li.className = "check";
    li.dataset.status = c.status;
    li.innerHTML = `<span class="check-icon" aria-hidden="true">${{ pass: "✓", warn: "!", fail: "✕" }[c.status]}</span><span class="check-name"></span><span class="check-value"></span>`;
    li.querySelector(".check-name").textContent = c.name;
    li.querySelector(".check-value").textContent = c.value;
    qList.appendChild(li);
  });

  /* ---------- Actions ---------- */
  const toast = $("toast");
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("is-visible");
    setTimeout(() => toast.classList.remove("is-visible"), 2200);
  }

  $("print-btn").addEventListener("click", () => window.print());

  $("copy-btn").addEventListener("click", async () => {
    const lines = [
      `Black Sigatoka Detector result`,
      `Photo: ${scan.fileName}`,
      `Analysed: ${when.toLocaleString()}`,
      `Result: ${outcome.title}`,
      `${outcome.meterLabel}: ${pct.toFixed(1)}%`,
      `Model: ${MODEL_INFO.activeName} (validation recall ${formatPct(m.recall)}, F1 ${formatPct(m.f1)})`,
      result.id ? `Reference: #${result.id}` : null,
      "This is a screening aid, not a laboratory diagnosis. Exotic Plant Pest Hotline: 1800 084 881.",
    ].filter(Boolean);
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      showToast("Summary copied");
    } catch (_) {
      showToast("Couldn't access the clipboard");
    }
  });
});
