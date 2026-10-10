/*
 * API layer for the Black Sigatoka Detector frontend.
 *
 * The real backend (backend/detector/views.py) returns the raw Prediction
 * model via PredictionSerializer:
 *   { id, image, label, confidence, is_known_disease, is_banana_leaf,
 *     heatmap, created_at }
 * where `label` is one of "black_sigatoka" | "healthy" | "not_black_sigatoka"
 * and `heatmap` is a relative URL to a GRAYSCALE Grad-CAM PNG (or "").
 *
 * normaliseResult() turns that into the shape the result page uses, so
 * nothing else in the frontend needs to know about the backend format.
 *
 * Demo tip: add ?mock=1 to the upload page URL to run without the backend,
 * or ?mock=black_sigatoka / ?mock=healthy / ?mock=not_black_sigatoka to
 * force a specific outcome.
 */

const API_BASE_URL = window.location.origin; // Django hosts both frontend and API

const MOCK_PARAM = new URLSearchParams(window.location.search).get("mock");
const USE_MOCK_API = MOCK_PARAM !== null;

/**
 * Sends an image file for prediction.
 * Resolves to { id, label, confidence, is_known_disease, heatmap_url, created_at, mocked }
 * or throws an Error with a user-facing message.
 */
async function predictImage(file) {
  const raw = USE_MOCK_API ? await mockPredict(MOCK_PARAM) : await realPredict(file);
  return normaliseResult(raw);
}

async function realPredict(file) {
  const formData = new FormData();
  formData.append("image", file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/predict/`, { method: "POST", body: formData });
  } catch (networkError) {
    throw new Error(
      `Couldn't reach the analysis server at ${API_BASE_URL}. ` +
      "Check that Django is running (python manage.py runserver), then try again."
    );
  }

  if (!response.ok) {
    let message = `The server couldn't analyse this image (error ${response.status}).`;
    try {
      const body = await response.json();
      if (body.error) message = body.error;
      else if (body.detail) message = body.detail;
    } catch (_) { /* not JSON — keep the default message */ }
    throw new Error(message);
  }
  return response.json();
}

function toAbsoluteUrl(path) {
  if (!path) return null;
  if (path.startsWith("data:") || path.startsWith("blob:")) return path;
  try { return new URL(path, API_BASE_URL).href; } catch (_) { return null; }
}

function normaliseResult(raw) {
  const label = ["black_sigatoka", "healthy", "not_black_sigatoka"].includes(raw.label)
    ? raw.label
    : "not_black_sigatoka";
  return {
    id: raw.id ?? null,
    label,
    confidence: Math.max(0, Math.min(1, Number(raw.confidence) || 0)),
    is_known_disease: raw.is_known_disease ?? label !== "not_black_sigatoka",
    heatmap_url: toAbsoluteUrl(raw.heatmap_data_url || raw.heatmap || raw.heatmap_url),
    heatmap_error: raw.heatmap_error || null,
    // Cosine-distance output from DINOv2 stage one. These are NOT percentages.
    gate_distance: raw.gate_distance == null ? null : Number(raw.gate_distance),
    gate_threshold: raw.gate_threshold == null ? null : Number(raw.gate_threshold),
    gate_nearest_class: raw.gate_nearest_class ?? null,
    classifier_key: raw.classifier_key ?? null,
    heatmap_overlay_url: toAbsoluteUrl(raw.heatmap_overlay_url),
    background_removed_applied: Boolean(raw.background_removed_applied),
    background_removal_method: raw.background_removal_method || null,
    background_removal_reason: raw.background_removal_reason || null,
    background_removed_preview_url: toAbsoluteUrl(raw.background_removed_preview_url),
    created_at: raw.created_at || new Date().toISOString(),
    mocked: Boolean(raw._mock),
  };
}

/* ---------------- Mock mode ---------------- */

function mockPredict(forced) {
  const labels = ["black_sigatoka", "healthy", "not_black_sigatoka"];
  const label = labels.includes(forced) ? forced : labels[Math.floor(Math.random() * 3)];
  return new Promise((resolve) => {
    setTimeout(() => {
      const known = label !== "not_black_sigatoka";
      resolve({
        _mock: true,
        id: null,
        label,
        confidence: known ? 0.78 + Math.random() * 0.2 : 0.45 + Math.random() * 0.2,
        is_known_disease: known,
        heatmap: known ? mockHeatmap(label) : "",
        created_at: new Date().toISOString(),
      });
    }, 1800);
  });
}

/* Grayscale 224×224 blob map — the same format inference.py produces. */
function mockHeatmap(label) {
  const size = 224;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  const blobs = label === "black_sigatoka"
    ? [[150, 70, 34], [175, 120, 26], [120, 150, 22]]
    : [[112, 112, 70]];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let v = 0;
      for (const [bx, by, r] of blobs) {
        v = Math.max(v, Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * r * r)));
      }
      const i = (y * size + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = Math.round(v * 255);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}
