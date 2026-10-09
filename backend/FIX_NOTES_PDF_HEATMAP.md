# PDF / Heatmap fixes (2026-10-10)

## Save as PDF
- The **Save as PDF** button now calls `window.print()` synchronously from
  the click, retaining browser permission to open a print dialog.
- An asynchronous canvas export can no longer block the button. The original
  and model-generated attention overlay are prepared separately. If the overlay
  isn't ready or fails, the remaining one-page report still prints.
- In the browser print dialog, select **Save as PDF**, paper **A4**, and
  ideally **Background graphics** (the stylesheet requests that setting).
- Headless Chromium tests verified an A4 one-page PDF for Black Sigatoka,
  Healthy and out-of-distribution responses; page content includes photo-quality
  findings, class scores and next steps. A simulated grayscale heatmap was
  included in the first two PDFs.

## More detailed attention
- `detector/cam.py`: Grad-CAM uses a spatial layer as a low-resolution prior,
  then refines it with absolute Gradient × Input sensitivity of the same
  predicted-class logit, without extra forward passes or retraining.
- `detector/heatmap.py` and `static/js/result.js`: quieter reds, cyan/yellow
  transitions and transparent low-attention regions.
- The image location is derived from class gradients, NOT from artificial
  colour segmentation. No guarantee of lesions or identical blobs to the
  reference. Please compare the output with real checkpoint-generated maps.
- The result page version-busts stylesheet/JS caching.

## Installation
Extract and replace the repository `backend/` folder (keep your existing
`db.sqlite3` and `ml/models` folder). From the `backend/` folder:

```
python manage.py runserver
```

Open http://127.0.0.1:8000/ , hard refresh with Ctrl+F5 if necessary, and
**scan the photo again**. Old sessionStorage results contain the *previous*
heatmap image; a new scan is required to regenerate the attention values.

Real-model validation is still required because pretrained weights are not
included in this update. This is a screening aid, not a lesion segmentation tool.
