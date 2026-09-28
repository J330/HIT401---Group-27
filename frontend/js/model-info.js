/*
 * Model evaluation figures shown on the Result and About pages.
 *
 * HOW TO FILL THIS IN (takes 2 minutes):
 *   1. cd ml && python evaluate.py
 *   2. For the model you set as ACTIVE_MODEL in backend/backend/settings.py,
 *      copy the 2×2 confusion matrix it prints into `confusion` below.
 *      sklearn prints rows = actual class, columns = predicted class, in
 *      alphabetical folder order: [black_sigatoka, healthy]. Paste as-is.
 *   3. Copy each model's accuracy from the "=== Summary ===" block into
 *      `candidates`.
 *   4. Set isPlaceholder to false.
 *
 * Accuracy, precision, recall, F1 and specificity are all CALCULATED from
 * the confusion matrix by model-metrics below, so they can never disagree
 * with each other.
 *
 * The numbers currently below are SAMPLE values sized to our real
 * validation split (129 Black Sigatoka / 277 healthy). While
 * isPlaceholder is true the site shows a visible "sample figures" notice.
 */

const MODEL_INFO = {
  isPlaceholder: true,

  activeKey: "convnextv2",               // must match ACTIVE_MODEL in settings.py
  activeName: "ConvNeXt V2 (Base, ImageNet-22k)",
  evaluatedOn: "Held-out validation split (20% of our cleaned dataset)",

  //            predicted: BS    healthy
  confusion: [ /* actual BS      */ [121, 8],
               /* actual healthy */ [6, 271] ],

  candidates: [
    { key: "convnextv2",      name: "ConvNeXt V2 (Base)",   accuracy: 0.9655 },
    { key: "efficientnetv2s", name: "EfficientNetV2-S",     accuracy: 0.9581 },
    { key: "swin",            name: "Swin Transformer (Base)", accuracy: 0.9532 },
    { key: "vit",             name: "ViT (Base, patch 16)", accuracy: 0.9409 },
  ],

  gate: {
    name: "DINOv2 open-set gate",
    threshold: 0.30,  // GATE_THRESHOLD in backend/detector/inference.py (cosine distance)
  },

  // How the classifier crops images before prediction, so the heatmap lines
  // up with the photo. ConvNeXt V2 resizes then centre-crops → "cover".
  // ViT and Swin resize straight to 224×224 → change this to "fill".
  heatmapFit: "cover",
};

/* Derived metrics — Black Sigatoka is treated as the positive class. */
function modelMetrics(info = MODEL_INFO) {
  const [[tp, fn], [fp, tn]] = info.confusion;
  const total = tp + fn + fp + tn;
  const safe = (n, d) => (d ? n / d : 0);
  const precision = safe(tp, tp + fp);
  const recall = safe(tp, tp + fn);
  return {
    tp, fn, fp, tn, total,
    positives: tp + fn,
    negatives: fp + tn,
    accuracy: safe(tp + tn, total),
    precision,
    recall,
    specificity: safe(tn, tn + fp),
    f1: safe(2 * precision * recall, precision + recall),
  };
}

function formatPct(x, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}
