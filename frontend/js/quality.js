/*
 * Client-side image quality checks, run before anything is uploaded.
 *
 * A "fail" blocks the upload; a "warn" lets it through with advice.
 *
 * Thresholds were calibrated on our own data/raw set (2,028 photos):
 *  - Sharpness = variance of the Laplacian, measured on the image scaled to
 *    at most 1024px, averaged over the sharpest quarter of a 6×6 tile grid
 *    (so a sharp leaf in front of a blurred background still passes).
 *    Measuring at 512px let heavily blurred 4K phone photos through, since
 *    shrinking hides blur; at 1024px real sharp phone photos in
 *    backend/media/uploads scored 80–1371 and blurred copies 3–38.
 *    Real training photos: 5th percentile 71, median 194.
 *    Same photos blurred by 1.5px: 95th percentile 18.
 *    Fail < 25 wrongly blocks only 0.7% of real training photos.
 *  - Minimum size 224px = the model's input size (also the smallest image
 *    in our training set).
 *  - Brightness/contrast limits are deliberately lenient because some of
 *    our training photos are shot on dark backgrounds.
 */

const QUALITY_RULES = {
  acceptedTypes: ["image/jpeg", "image/png", "image/webp"],
  maxBytes: 15 * 1024 * 1024,
  minShortSide: 224,
  goodShortSide: 400,
  sharpFail: 25,
  sharpWarn: 50,
  darkFail: 18,
  darkWarn: 40,
  brightFail: 238,
  brightWarn: 215,
  contrastFail: 8,
  contrastWarn: 16,
};

async function decodeImage(file) {
  if ("createImageBitmap" in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch (_) { /* fall back below */ }
  }
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("decode")); };
    img.src = url;
  });
}

function drawScaled(source, maxSide) {
  const w = source.width, h = source.height;
  const s = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(w * s));
  canvas.height = Math.max(1, Math.round(h * s));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high"; // closest to the resampling used in calibration
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return { canvas, ctx };
}

function measure(ctx, w, h) {
  const { data } = ctx.getImageData(0, 0, w, h);
  const gray = new Float32Array(w * h);
  let sum = 0, sumSq = 0, clipped = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const v = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    gray[p] = v; sum += v; sumSq += v * v;
    if (v > 250) clipped++;
  }
  const n = w * h;
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));

  // Laplacian variance per tile, then mean of the sharpest 25% of tiles
  const G = 6, tiles = [];
  for (let ty = 0; ty < G; ty++) {
    for (let tx = 0; tx < G; tx++) {
      const x0 = Math.max(1, Math.floor((tx * w) / G)), x1 = Math.min(w - 1, Math.floor(((tx + 1) * w) / G));
      const y0 = Math.max(1, Math.floor((ty * h) / G)), y1 = Math.min(h - 1, Math.floor(((ty + 1) * h) / G));
      let s = 0, s2 = 0, c = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const p = y * w + x;
          const lap = gray[p - 1] + gray[p + 1] + gray[p - w] + gray[p + w] - 4 * gray[p];
          s += lap; s2 += lap * lap; c++;
        }
      }
      if (c) tiles.push(s2 / c - (s / c) ** 2);
    }
  }
  tiles.sort((a, b) => b - a);
  const k = Math.max(1, Math.floor(tiles.length / 4));
  const sharpness = tiles.slice(0, k).reduce((a, b) => a + b, 0) / k;

  return { mean, std, sharpness, clippedRatio: clipped / n };
}

/**
 * Returns { ok, hasWarnings, checks[], width, height, previewDataUrl }
 * Each check: { id, name, status: "pass"|"warn"|"fail", value, message }
 */
async function assessImage(file) {
  const R = QUALITY_RULES;
  const checks = [];
  const add = (id, name, status, value, message) => checks.push({ id, name, status, value, message });

  const typeOk = R.acceptedTypes.includes(file.type);
  add("type", "File type", typeOk ? "pass" : "fail",
    (file.type.split("/")[1] || "unknown").toUpperCase(),
    typeOk ? "JPG, PNG or WebP." : "Use a JPG, PNG or WebP photo. iPhone HEIC photos need converting first.");

  const mb = file.size / (1024 * 1024);
  add("size", "File size", file.size <= R.maxBytes ? "pass" : "fail", `${mb.toFixed(1)} MB`,
    file.size <= R.maxBytes ? "Within the 15 MB limit." : "Larger than 15 MB. Export a smaller copy and try again.");

  if (!typeOk) return finish();

  let bitmap;
  try {
    bitmap = await decodeImage(file);
  } catch (_) {
    add("decode", "Readable image", "fail", "—", "This file couldn't be opened as an image. It may be damaged.");
    return finish();
  }

  const width = bitmap.width, height = bitmap.height;
  const shortSide = Math.min(width, height);
  add("resolution", "Resolution",
    shortSide < R.minShortSide ? "fail" : shortSide < R.goodShortSide ? "warn" : "pass",
    `${width} × ${height}`,
    shortSide < R.minShortSide
      ? `Too small — the shortest side must be at least ${R.minShortSide}px. Use the original photo, not a thumbnail.`
      : shortSide < R.goodShortSide
        ? "Usable, but a larger original will show lesions more clearly."
        : "Plenty of detail.");

  const { canvas, ctx } = drawScaled(bitmap, 1024);
  const m = measure(ctx, canvas.width, canvas.height);

  const sharp = Math.round(m.sharpness);
  add("sharpness", "Sharpness",
    sharp < R.sharpFail ? "fail" : sharp < R.sharpWarn ? "warn" : "pass",
    `score ${sharp}`,
    sharp < R.sharpFail
      ? "Too blurry to analyse. Hold the camera steady, tap the leaf to focus, and retake."
      : sharp < R.sharpWarn
        ? "Slightly soft. It will be analysed, but a sharper photo gives a more reliable result."
        : "In focus.");

  const lum = Math.round(m.mean);
  let expStatus = "pass", expMsg = "Well exposed.";
  if (lum < R.darkFail) { expStatus = "fail"; expMsg = "Far too dark. Photograph the leaf in daylight."; }
  else if (lum > R.brightFail) { expStatus = "fail"; expMsg = "Washed out. Avoid direct glare on the leaf surface."; }
  else if (lum < R.darkWarn) { expStatus = "warn"; expMsg = "Quite dark. Lesions may be hard to see."; }
  else if (lum > R.brightWarn || m.clippedRatio > 0.25) { expStatus = "warn"; expMsg = "Some areas are overexposed. Try shade or an overcast moment."; }
  add("exposure", "Lighting", expStatus, `brightness ${lum}/255`, expMsg);

  const con = Math.round(m.std);
  add("contrast", "Contrast",
    con < R.contrastFail ? "fail" : con < R.contrastWarn ? "warn" : "pass",
    `${con}`,
    con < R.contrastFail
      ? "Almost no detail — the image looks flat or blank."
      : con < R.contrastWarn ? "Low contrast. Fog, haze or a dirty lens can cause this." : "Good detail range.");

  // Downscaled copy for the result page (keeps sessionStorage under its ~5 MB limit)
  const { canvas: small } = drawScaled(bitmap, 1024);
  const previewDataUrl = small.toDataURL("image/jpeg", 0.88);
  if (bitmap.close) bitmap.close();

  return finish({ width, height, previewDataUrl });

  function finish(extra = {}) {
    return {
      ok: !checks.some((c) => c.status === "fail"),
      hasWarnings: checks.some((c) => c.status === "warn"),
      checks,
      ...extra,
    };
  }
}
