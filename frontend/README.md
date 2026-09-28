# Black Sigatoka Detector: Frontend

Plain HTML/CSS/JS. No build step, no framework.

## Run it

Backend running on port 8000 first (see the main README), then in a second terminal:

```
cd frontend
python -m http.server 8080
```

Open **http://127.0.0.1:8080/** — use `127.0.0.1`, not `localhost`. Django's
`CORS_ALLOWED_ORIGINS` only allows `http://127.0.0.1:8080`, so `localhost:8080`
will fail with a "couldn't reach the server" error.

### Demo without the backend

Add `?mock=` to the upload page URL:

- `upload.html?mock=1` random outcome
- `upload.html?mock=black_sigatoka`
- `upload.html?mock=healthy`
- `upload.html?mock=not_black_sigatoka`

Mock results are labelled "demo result" on the report.

## Before you demo: add the real evaluation numbers

The result page shows accuracy, recall, precision, F1, specificity and a
confusion matrix. The backend doesn't store these, so they live in
`js/model-info.js`. It currently holds **sample figures** and the site says so.

1. `cd ml && python evaluate.py`
2. Copy the 2×2 confusion matrix printed for your `ACTIVE_MODEL` into
   `confusion` (paste it exactly as sklearn prints it).
3. Copy each model's accuracy from the Summary block into `candidates`.
4. Set `isPlaceholder: false`.

All five metrics are calculated from the confusion matrix, so they always agree.
If you switch `ACTIVE_MODEL` to `vit` or `swin`, also set `heatmapFit: "fill"`
so the heatmap lines up with the photo.

## What's in it

| File | Purpose |
|---|---|
| `index.html` | Home |
| `upload.html` | Photo picker, camera, drag-drop, paste, quality checks |
| `result.html` | Leaf report |
| `about.html` | Pipeline, model comparison, biosecurity, limitations |
| `css/style.css` | All styling |
| `js/api.js` | Calls `/api/predict/`, adapts the response, mock mode |
| `js/quality.js` | Blur / resolution / lighting / contrast checks |
| `js/model-info.js` | Evaluation figures (fill this in) |
| `js/upload.js`, `js/result.js` | Page logic |

### Photo quality checks (`js/quality.js`)

Run in the browser before upload. A failed check blocks the upload; a
warning lets it through with advice. Thresholds were calibrated on our own
`data/raw` photos: the blur limit wrongly blocks 0.7% of real training
photos while catching every photo blurred by 1.5px or more. Tune the numbers
in `QUALITY_RULES` if needed.

### Result page

- Verdict for all three backend labels, with next steps (hotline 1800 084 881)
- Confidence meter with a High/Moderate/Low band. For `not_black_sigatoka`
  it shows gate similarity against the 70% threshold instead
- Grad-CAM viewer: Photo / Overlay / Heatmap views, strength slider,
  hover-to-read attention values, strongest-focus location
- Model reliability: 5 metrics in plain language, confusion matrix,
  "could this be wrong?" explanations
- Save as PDF (print layout) and Copy summary

### Changes from the previous version

- `api.js` now matches the real backend response (`label`, `confidence`,
  relative `heatmap` URL) instead of the old `label_display`/`heatmap_url`
  mock shape. The backend's heatmap is grayscale; the frontend colours it.
- Mock mode is off by default and turned on per-visit with `?mock=`.
- The photo passed to the result page is downscaled first. The old version
  stored the full-size image in sessionStorage, which overflows its ~5 MB
  limit on most phone photos.
- About page methodology updated to the models actually in `ml/`.
- Footer changed from HIT237 to HIT401.
