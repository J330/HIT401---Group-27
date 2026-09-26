/* Render the two-stage DINOv2 result in a clearer evidence-first layout. */

document.addEventListener("DOMContentLoaded", () => {
  const raw = sessionStorage.getItem("lastScanResult");
  const image = sessionStorage.getItem("lastScanImage");

  if (!raw || !image) {
    window.location.href = window.UPLOAD_URL || "/upload/";
    return;
  }

  const result = JSON.parse(raw);
  const initial = result.initial_scan;
  const health = result.health_scan;

  if (!initial) {
    window.location.href = window.UPLOAD_URL || "/upload/";
    return;
  }

  document.getElementById("result-image").src = image;
  document.getElementById("heatmap-original").src = image;

  renderStage1(initial);

  const healthCard = document.getElementById("health-scan-card");
  const rejectionCard = document.getElementById("rejection-card");
  const heatmapSection = document.getElementById("heatmap-section");
  const heatmapErrorSection = document.getElementById("heatmap-error-section");

  if (!result.accepted || !health) {
    healthCard.style.display = "none";
    rejectionCard.style.display = "block";
    heatmapSection.style.display = "none";
    heatmapErrorSection.style.display = "none";
    return;
  }

  healthCard.style.display = "block";
  rejectionCard.style.display = "none";
  renderStage2(health, result);
  renderHeatmap(health, image);
});

function renderStage1(initial) {
  const bananaSimilarity = Number(
    initial.banana_leaf_similarity ?? initial.banana_leaf_probability ?? 0
  );
  const notBananaSimilarity = Number(initial.not_banana_leaf_similarity ?? 0);
  const threshold = Number(initial.threshold ?? 0.65);
  const requiredMargin = Number(initial.required_margin ?? 0);

  const bananaPct = percent(bananaSimilarity);
  const notBananaPct = percent(notBananaSimilarity);
  const thresholdPct = percent(threshold);
  const marginPct = percent(requiredMargin);

  const label = document.getElementById("initial-label");
  label.textContent = initial.label_display ||
    (initial.accepted ? "Banana Leaf" : "Not Banana Leaf");
  label.classList.add(initial.accepted ? "negative" : "positive");

  const pill = document.getElementById("initial-status-pill");
  setStatusPill(pill, initial.accepted, initial.accepted ? "Passed" : "Rejected");

  document.getElementById("banana-score-value").textContent = `${bananaPct}%`;
  document.getElementById("not-banana-score-value").textContent = `${notBananaPct}%`;
  document.getElementById("banana-threshold-value").textContent = `${thresholdPct}%`;
  document.getElementById("banana-confidence-fill").style.width = `${bananaPct}%`;

  document.getElementById("threshold-text").textContent =
    `Banana Leaf must reach ${thresholdPct}% similarity and lead Not Banana Leaf by at least ${marginPct}%.`;

  renderProbabilities(
    "initial-probability-list",
    initial.similarity_scores || initial.probabilities || {},
    false
  );
}

function renderStage2(health, result) {
  document.getElementById("result-model").textContent =
    health.model_display || result.model_display || "Health AI";

  const finalLabel = health.label_display || "Unknown";
  const labelEl = document.getElementById("result-label");
  labelEl.textContent = finalLabel;
  labelEl.classList.add(health.label === "healthy" ? "negative" : "positive");

  const reliable = health.reliable_match !== false && health.label !== "unknown";
  const pill = document.getElementById("health-status-pill");
  setStatusPill(pill, reliable, reliable ? "Reliable match" : "Not accepted");

  const finalBox = document.getElementById("final-result-box");
  finalBox.classList.toggle("final-result-rejected", !reliable);
  finalBox.classList.toggle("final-result-accepted", reliable);

  const classifierConfidence = Number(
    health.classifier_confidence ?? health.confidence ?? 0
  );
  const classifierLabel =
    health.classifier_label_display || health.label_display || "Unknown";

  document.getElementById("classifier-choice-value").textContent = classifierLabel;
  document.getElementById("classifier-confidence-value").textContent =
    `${percent(classifierConfidence)}% classifier confidence`;

  renderProbabilities("probability-list", health.probabilities || {}, true);

  const similarity = health.similarity;
  const similarityDetails = document.getElementById("similarity-details");
  const referenceCard = document.getElementById("reference-evidence-card");
  const decisionPanel = document.getElementById("decision-panel");

  if (similarity && similarity.scores) {
    similarityDetails.style.display = "block";
    referenceCard.style.display = "flex";
    decisionPanel.style.display = "block";

    const bestScore = Number(similarity.best_score || 0);
    const margin = Number(similarity.margin || 0);
    const threshold = Number(similarity.threshold || 0);
    const requiredMargin = Number(similarity.required_margin || 0);
    const bestClass = prettyLabel(similarity.best_class || "Unknown");

    document.getElementById("reference-choice-value").textContent = bestClass;
    document.getElementById("reference-score-value").textContent =
      `${formatSimilarity(bestScore)} visual similarity`;

    document.getElementById("similarity-summary").textContent =
      `Closest verified class: ${bestClass} (${formatSimilarity(bestScore)})`;
    renderProbabilities("similarity-list", similarity.scores, false);
    document.getElementById("similarity-threshold-text").textContent =
      `Required visual similarity: ${formatSimilarity(threshold)}`;
    document.getElementById("similarity-margin-text").textContent =
      `Best-vs-second margin: ${formatSimilarity(margin)}; required: ${formatSimilarity(requiredMargin)}`;

    renderDecisionChecks(health, similarity, classifierConfidence);
  } else {
    similarityDetails.style.display = "none";
    referenceCard.style.display = "none";
    decisionPanel.style.display = "none";
  }

  const explanation = document.getElementById("final-result-explanation");
  if (!reliable) {
    explanation.textContent =
      "The classifier made a strong choice, but one or more reliability checks failed, so the system did not force a disease label.";
  } else {
    explanation.textContent =
      "The classifier and reference-similarity checks passed the configured acceptance rules.";
  }

  renderModelMetrics(health.model_metrics);

  const nextSteps = document.getElementById("next-steps");
  if (!reliable) {
    nextSteps.textContent =
      "Try another clear image of the same leaf if you want a second screening result. Unknown means the evidence was not strong and consistent enough to accept a disease class.";
  } else if (health.label === "black_sigatoka") {
    nextSteps.textContent =
      "The AI screening supports Black Sigatoka. Use this as a screening result rather than a confirmed diagnosis.";
  } else if (health.label === "healthy") {
    nextSteps.textContent =
      "The AI screening supports the Healthy class. If visible symptoms remain, scan another clear area of the leaf.";
  } else {
    nextSteps.textContent =
      "The classifier and reference check agreed sufficiently for this class. Treat the output as AI screening rather than a confirmed diagnosis.";
  }
}

function renderDecisionChecks(health, similarity, classifierConfidence) {
  const list = document.getElementById("decision-check-list");
  list.innerHTML = "";

  const checks = health.decision_checks || {};
  const items = [
    {
      label: "Classifier and closest reference agree",
      pass: checks.classifier_reference_agreement,
    },
    {
      label: `Reference similarity reaches ${formatSimilarity(similarity.threshold || 0)}`,
      pass: checks.similarity_threshold_pass,
    },
    {
      label: `Best class leads second place by ${formatSimilarity(similarity.required_margin || 0)}`,
      pass: checks.similarity_margin_pass,
    },
    {
      label: `Classifier confidence reaches ${percent(similarity.classifier_min_confidence || 0)}%`,
      pass: checks.classifier_confidence_pass,
    },
  ];

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = `decision-check ${item.pass ? "decision-pass" : "decision-fail"}`;

    const icon = document.createElement("span");
    icon.className = "decision-icon";
    icon.textContent = item.pass ? "✓" : "×";

    const text = document.createElement("span");
    text.textContent = item.label;

    row.append(icon, text);
    list.appendChild(row);
  });

  const reasons = document.getElementById("rejection-reasons");
  reasons.innerHTML = "";
  if (health.reliable_match === false && (health.rejection_reasons || []).length) {
    const heading = document.createElement("strong");
    heading.textContent = "Why the final result was Unknown:";
    reasons.appendChild(heading);

    const ul = document.createElement("ul");
    health.rejection_reasons.forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      ul.appendChild(li);
    });
    reasons.appendChild(ul);
  }
}

function renderModelMetrics(metrics) {
  const panel = document.getElementById("metrics-panel");
  const grid = document.getElementById("metrics-grid");
  const note = document.getElementById("metrics-note");
  const testSize = document.getElementById("metrics-test-size");
  const perClassDetails = document.getElementById("per-class-metrics-details");
  const perClassBody = document.getElementById("per-class-metrics-body");

  if (!metrics || typeof metrics !== "object") {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "block";
  grid.innerHTML = "";

  const metricItems = [
    ["Accuracy", metrics.accuracy],
    ["Balanced accuracy", metrics.balanced_accuracy],
    ["Macro precision", metrics.macro_precision],
    ["Macro recall", metrics.macro_recall],
    ["Macro F1", metrics.macro_f1],
    ["Weighted F1", metrics.weighted_f1],
  ].filter(([, value]) => value !== undefined && value !== null);

  metricItems.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("strong");
    number.textContent = `${percent(Number(value))}%`;
    card.append(name, number);
    grid.appendChild(card);
  });

  testSize.textContent = metrics.test_size
    ? `${metrics.test_size} test images`
    : "";

  note.textContent = metrics.note ||
    (metrics.source === "checkpoint"
      ? "Accuracy was stored in the training checkpoint. Run the evaluation script once to add F1, precision and recall."
      : "Statistics are measured on the held-out test set, not on this individual uploaded image.");

  const perClass = metrics.per_class;
  if (perClass && typeof perClass === "object") {
    perClassBody.innerHTML = "";
    Object.entries(perClass).forEach(([name, row]) => {
      const tr = document.createElement("tr");
      const values = [
        prettyLabel(name),
        `${percent(row.precision)}%`,
        `${percent(row.recall)}%`,
        `${percent(row.f1)}%`,
        String(row.support ?? "—"),
      ];
      values.forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      perClassBody.appendChild(tr);
    });
    perClassDetails.style.display = "block";
  } else {
    perClassDetails.style.display = "none";
  }
}

function renderHeatmap(health, image) {
  const heatmapSection = document.getElementById("heatmap-section");
  const errorSection = document.getElementById("heatmap-error-section");
  document.getElementById("heatmap-original").src = image;

  if (health.heatmap_url) {
    document.getElementById("heatmap-image").src = health.heatmap_url;
    document.getElementById("heatmap-method").textContent =
      health.heatmap_method || "Grad-CAM";

    const target = prettyLabel(
      health.heatmap_target || health.classifier_label_display || "prediction"
    );
    document.getElementById("heatmap-target").textContent =
      health.reliable_match === false
        ? `Heatmap target: ${target}. The classifier choice was rejected by the reliability checks.`
        : `Heatmap generated for the accepted class: ${target}`;

    heatmapSection.style.display = "block";
    errorSection.style.display = "none";
  } else {
    heatmapSection.style.display = "none";
    if (health.heatmap_error) {
      document.getElementById("heatmap-error-text").textContent =
        ` ${health.heatmap_error}`;
      errorSection.style.display = "block";
    } else {
      errorSection.style.display = "none";
    }
  }
}

function setStatusPill(element, pass, text) {
  element.textContent = text;
  element.classList.remove("status-pass", "status-fail");
  element.classList.add(pass ? "status-pass" : "status-fail");
}

function percent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  const normalised = number > 1 ? number / 100 : number;
  return Math.round(Math.max(0, Math.min(1, normalised)) * 100);
}

function formatSimilarity(value) {
  return `${percent(value)}%`;
}

function prettyLabel(value) {
  return String(value || "Unknown").replaceAll("_", " ");
}

function renderProbabilities(targetId, values, asProbability) {
  const list = document.getElementById(targetId);
  list.innerHTML = "";

  Object.entries(values)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "probability-row";

      const name = document.createElement("span");
      name.textContent = prettyLabel(label);

      const number = document.createElement("strong");
      number.textContent = asProbability
        ? `${percent(value)}%`
        : formatSimilarity(value);

      row.append(name, number);
      list.appendChild(row);
    });
}
