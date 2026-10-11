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

// Colormap: blue/cyan rims, yellow transitions and intense red hotspots.
// Positions match detector/heatmap.py so the API overlay looks the same.
const HOTSPOT_STOPS = [
  [0.00, 8, 24, 97], [0.24, 10, 76, 222], [0.37, 0, 190, 255],
  [0.51, 30, 245, 183], [0.64, 255, 238, 28], [0.78, 255, 131, 8],
  [0.91, 255, 29, 12], [1.00, 196, 0, 0],
];

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
  // Printing must remain a synchronous, direct user action. A failed canvas
  // export or an unfinished heatmap must never prevent saving the PDF.

  function setMode(mode) {
    stage.dataset.mode = mode;
    modeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.modeBtn === mode)));
    $("opacity-row").hidden = mode !== "overlay";
    paintHeatmap();
  }
  modeButtons.forEach((b) => b.addEventListener("click", () => {
    if (!b.disabled) setMode(b.dataset.modeBtn);
  }));
  opacity.addEventListener("input", () => {
    opacityOut.textContent = `${opacity.value}%`;
    paintHeatmap();
  });

  if (result.heatmap_url) {
    loadHeatmap(result.heatmap_url).then(() => {
      const heatmapButton = modeButtons.find((b) => b.dataset.modeBtn === "overlay");
      if (heatmapButton) {
        heatmapButton.disabled = false;
        heatmapButton.title = "View AI attention heatmap";
      }
      setMode("overlay");
      updatePrintAttention();
      // Prepare the PDF preview independently: its failure must not block Save as PDF.
      capturePrintOverlay().catch((error) => console.warn("PDF overlay fallback:", error));
    }).catch((error) => {
      console.error("Failed to render heatmap:", error);
      noHeatmap(error.message);
    });
  } else {
    noHeatmap();
  }

  function noHeatmap(viewerError) {
    modeButtons.forEach((b) => { if (b.dataset.modeBtn !== "photo") b.disabled = true; });
    setMode("photo");
    $("opacity-row").hidden = true;
    $("legend").hidden = true;
    $("heatmap-stats").hidden = true;
    const msg = $("viewer-empty");
    msg.hidden = false;
    msg.textContent = result.label === "not_black_sigatoka"
      ? "No attention map: the classifier didn't run because the photo was outside what the model knows."
      : (result.heatmap_error || viewerError || "The attention map couldn't be generated for this photo.") + " The diagnosis above is unaffected.";
    $("print-heatmap-status").textContent = msg.textContent;
    $("print-focus-peak").textContent = "Unavailable";
    $("print-focus-area").textContent = "Unavailable";
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
      if (camValues[i] >= 0.6) focus++;
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
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      // Let the original leaf show through unimportant pixels.
      // Blue/cyan form the boundary of a hotspot; orange/red mark its core.
      // Keep cyan/yellow transition rings visible; reduce the large, solid
      // red patches that previously covered the underlying photograph.
      const t = Math.max(0, Math.min(1, (v - 0.19) / 0.76));
      const fade = t * t * (3 - 2 * t);
      const alpha = v < 0.19 ? 0 : (0.10 + 0.85 * fade);
      img.data[o + 3] = overlay ? Math.round(255 * alphaScale * alpha) : 245;
    }
    ctx.putImageData(img, 0, 0);
  }

  function colormap(v) {
    const value = Math.max(0, Math.min(1, v));
    for (let i = 1; i < HOTSPOT_STOPS.length; i++) {
      const last = HOTSPOT_STOPS[i - 1];
      const next = HOTSPOT_STOPS[i];
      if (value <= next[0]) {
        const f = (value - last[0]) / (next[0] - last[0]);
        return [1, 2, 3].map((ch) => last[ch] + (next[ch] - last[ch]) * f);
      }
    }
    return HOTSPOT_STOPS[HOTSPOT_STOPS.length - 1].slice(1);
  }

  // Hover / touch readout of attention at a point
  function probeAt(clientX, clientY) {
    if (!camValues || stage.dataset.mode === "photo") { probe.style.display = "none"; return; }
    const rect = stage.getBoundingClientRect();
    const stageX = clientX - rect.left;
    const stageY = clientY - rect.top;
    if (stageX < 0 || stageX > rect.width || stageY < 0 || stageY > rect.height) {
      probe.style.display = "none";
      return;
    }
    const [w, h] = camSize;
    let x, y;
    if (stage.dataset.fit === "fill") {
      x = stageX / rect.width;
      y = stageY / rect.height;
    } else {
      // Match the photo AND the heatmap's object-fit: cover crop.
      const scale = Math.max(rect.width / w, rect.height / h);
      const cropX = (w * scale - rect.width) / 2;
      const cropY = (h * scale - rect.height) / 2;
      x = (stageX + cropX) / (w * scale);
      y = (stageY + cropY) / (h * scale);
    }
    const px = Math.min(w - 1, Math.floor(x * w));
    const py = Math.min(h - 1, Math.floor(y * h));
    const v = camValues[py * w + px];
    probe.textContent = `Relative attention ${(v * 100).toFixed(0)}%`;
    probe.style.left = `${(stageX / rect.width) * 100}%`;
    probe.style.top = `${(stageY / rect.height) * 100}%`;
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

  /* ---------- Detailed single-page PDF report ---------- */
  $("print-meta").textContent = $("report-meta").textContent;
  $("print-verdict").textContent = outcome.title;
  $("print-summary").textContent = outcome.summary;
  $("print-confidence-label").textContent = outcome.meterLabel;
  $("print-confidence-value").textContent = `${pct.toFixed(1)}%`;
  $("print-demo").hidden = !result.mocked;
  $("print-image").src = scan.image || "";
  if (!scan.image) $("print-image").alt = "Leaf photo unavailable";
  function showPrintHeatmap(url) {
    if (!url) return;
    $("print-heatmap-image").src = url;
    $("print-heatmap-image").hidden = false;
    $("print-heatmap-fallback").hidden = true;
  }
  // The API supplies an already aligned overlay; use it immediately when present.
  showPrintHeatmap(result.heatmap_overlay_url);

  const classified = result.label !== "not_black_sigatoka";
  const gateDistance = Number(result.gate_distance);
  const gateThreshold = Number(result.gate_threshold);
  const hasGateNumbers = result.gate_distance != null && Number.isFinite(gateDistance)
    && result.gate_threshold != null && Number.isFinite(gateThreshold);
  const gateClass = result.gate_nearest_class
    ? String(result.gate_nearest_class).replaceAll("_", " ") : null;
  const gateStatement = hasGateNumbers
    ? `${classified ? "Passed" : "Rejected"}: cosine distance ${gateDistance.toFixed(3)} (limit ${gateThreshold.toFixed(3)}).${gateClass ? ` Closest training category: ${gateClass}.` : ""}`
    : `${classified ? "Passed" : "Not passed"}; numeric distance not recorded for this scan.`;
  $("print-gate").textContent = gateStatement;
  $("print-gate-explanation").textContent = hasGateNumbers
    ? "DINOv2 measures similarity to training examples using cosine distance; lower distances mean greater similarity. This is not a calibrated banana-leaf probability."
    : "DINOv2 filters images that do not closely resemble the model's known training categories.";

  const activeCandidate = (MODEL_INFO.candidates || []).find((model) => model.key === result.classifier_key);
  const configuredModel = activeCandidate?.name || MODEL_INFO.activeName;
  $("print-classifier").textContent = classified
    ? `${configuredModel}; binary prediction: Black Sigatoka vs Healthy.`
    : "Not run because the similarity gate rejected the image.";
  $("print-confidence-explainer").textContent = classified
    ? "Classifier output for this image; not disease severity."
    : "Similarity score used for gate screening; not diagnostic confidence.";
  if (classified) {
    const blackSigatoka = result.label === "black_sigatoka" ? result.confidence : 1 - result.confidence;
    const healthy = 1 - blackSigatoka;
    $("print-prob-bs").style.width = `${(blackSigatoka * 100).toFixed(2)}%`;
    $("print-prob-healthy").style.width = `${(healthy * 100).toFixed(2)}%`;
    $("print-prob-bs-text").textContent = formatPct(blackSigatoka);
    $("print-prob-healthy-text").textContent = formatPct(healthy);
  } else {
    $("print-probabilities").hidden = true;
  }

  // Actual pre-upload measurements: don't infer diseases from these checks.
  const quality = Array.isArray(scan.quality) ? scan.quality : [];
  const warningChecks = quality.filter((q) => q.status === "warn" || q.status === "fail");
  $("print-quality-overview").textContent = quality.length
    ? `${quality.length} checks completed · ${warningChecks.length} flagged. Photo quality influences reliability, not diagnosis.`
    : "No recorded photo-quality checks for this scan.";
  quality.forEach((q) => {
    const row = document.createElement("div");
    row.className = `print-quality-row print-quality-${["warn", "fail"].includes(q.status) ? q.status : "pass"}`;
    const name = document.createElement("span");
    name.textContent = q.name || "Check";
    const reading = document.createElement("strong");
    reading.textContent = q.value || "—";
    const status = document.createElement("em");
    status.textContent = (q.status || "unknown").toUpperCase();
    row.append(name, reading, status);
    $("print-quality-list").appendChild(row);
  });
  // List every flagged measurement, but limit prose advice so one page still
  // fits for unusually poor photos with several simultaneous warnings.
  $("print-quality-advice").textContent = warningChecks.length
    ? `Flags: ${warningChecks.map((q) => `${q.name} (${q.value || "review"})`).join(", ")}. ` +
      `Guidance: ${warningChecks[0].message || "Retake the photo in clearer conditions."}`
    : "No quality warnings were recorded for this image.";

  // This value comes from the actual loaded Grad-CAM; no heatmap is invented.
  $("print-heatmap-status").textContent = result.heatmap_url
    ? "Gradient-guided Grad-CAM loading. The map highlights relative influence on the predicted class."
    : "No attention map is available for this scan.";
  function updatePrintAttention() {
    if (!camValues) return;
    $("print-heatmap-status").textContent = "Gradient-guided Grad-CAM for the predicted class. Red/yellow areas show stronger relative attention; cyan marks transition areas.";
    $("print-focus-peak").textContent = $("stat-peak").textContent;
    $("print-focus-area").textContent = $("stat-focus").textContent + " at relative attention ≥ 60%";
  }
  async function capturePrintOverlay() {
    if (!camValues || !photo.src) return;
    if (!photo.complete) {
      try { await photo.decode(); } catch (_) { return; }
    }
    if (!photo.naturalWidth) return;
    const [w, h] = camSize;
    const composite = document.createElement("canvas");
    composite.width = w; composite.height = h;
    const ctx = composite.getContext("2d");
    ctx.drawImage(photo, 0, 0, w, h);
    // Canvas is the same size as the CAM; honours the site's current colour map.
    ctx.drawImage(canvas, 0, 0, w, h);
    showPrintHeatmap(composite.toDataURL("image/jpeg", 0.89));
  }

  // Never print sample evaluation numbers as if they were real results.
  const verifiedMetrics = !MODEL_INFO.isPlaceholder &&
    (!result.classifier_key || result.classifier_key === MODEL_INFO.activeKey);
  if (verifiedMetrics) {
    $("print-eval-badge").textContent = "VALIDATION DATA";
    const figures = [
      ["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall],
      ["F1 score", m.f1], ["Specificity", m.specificity],
    ];
    const wrap = document.createElement("div");
    wrap.className = "print-metric-grid";
    figures.forEach(([label, value]) => {
      const el = document.createElement("div");
      const title = document.createElement("span"); title.textContent = label;
      const number = document.createElement("strong"); number.textContent = formatPct(value);
      el.append(title, number); wrap.appendChild(el);
    });
    $("print-performance-content").appendChild(wrap);
    $("print-performance-note").textContent = `${MODEL_INFO.evaluatedOn}: ${m.total} images (${m.positives} Black Sigatoka, ${m.negatives} healthy). Errors: ${m.fn} missed diseased leaves, ${m.fp} false alarms. These are aggregate test figures, not this photo's accuracy.`;
  } else {
    $("print-eval-badge").textContent = "NOT VERIFIED";
    const empty = document.createElement("p");
    empty.className = "print-unverified";
    empty.textContent = "Verified accuracy, precision, recall, F1 and specificity figures have not been provided for the active classifier. No sample figures are used in this PDF.";
    $("print-performance-content").appendChild(empty);
    $("print-performance-note").textContent = "Use ml/evaluate.py to produce results on a held-out dataset, then enter the verified confusion matrix in static/js/model-info.js.";
  }

  const printSteps = $("print-steps-list");
  outcome.steps.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    printSteps.appendChild(li);
  });
  $("print-interpretation").textContent = classified
    ? `The classifier favoured ${result.label === "black_sigatoka" ? "Black Sigatoka" : "Healthy"} among its two known labels. A ${pct.toFixed(1)}% score does not prove a diagnosis or rule out other leaf diseases.`
    : "The DINOv2 gate flagged this as unlike its known examples, so the disease classifier did not run. This is not proof the leaf is disease-free.";

  $("print-btn").addEventListener("click", () => {
    // Call window.print() directly from the click event. Browsers may ignore
    // print dialogs when invoked after async awaits. The complete report and
    // its server-generated overlay were populated before this handler ran.
    window.print();
  });

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
