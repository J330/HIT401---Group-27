# Leaf-only heatmap mask update

Previously rembg produced a grey-background leaf preview, but its transparent alpha mask
was discarded. The CAM colour overlay therefore painted the entire rectangular image.

This update retains U2Net's foreground alpha in `detector/background_filter.py` alongside
the identically cropped preview and passes it through the scan's Grad-CAM path.

- Stage 1 similarity and Stage 2 disease classification STILL use the original photo.
- The explanation's second forward pass uses the isolated leaf (as before).
- Grad-CAM display enhancement is normalized within the foreground and local red expansion
  is prevented from escaping the foreground boundary.
- The API outputs an RGBA heatmap PNG: red/green/blue channels carry grayscale CAM
  values and the alpha channel carries a softly feathered silhouette.
- The browser composites heatmap colours only where alpha is nonzero; hovering outside
  reads "Background excluded". Heatmap statistics refer to visible leaf pixels.
- The server's report/PDF image is masked by exactly the same silhouette.
- The existing sensitive red colour mapping is retained **inside** the leaf.
- If rembg fails, the original-image full-frame heatmap continues as a fallback;
  there is no invented leaf mask.

Soft masking and hotspot growth are visualization aids only: Grad-CAM is not a
Sigatoka lesion segmentation and the isolation-only explanation can differ from
actual attribution on the original photograph.

## Install / verify

Copy the updated `backend/` folder contents into the existing project, keeping
trained checkpoints and settings you store outside this ZIP. Install dependencies
with `pip install -r requirements.txt`, restart the Django server and scan a
NEW image (old results in sessionStorage cannot contain the new alpha mask).

Tests: `python -m unittest discover -s tests -v`

## Compatibility fix for disabled Heatmap button

The prior build imported `detector/heatmap.py` only after completing the
prediction. Under Python 3.9, annotations like `Image.Image | None` raised
`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` during
that import. Therefore the API returned no heatmap URL and the browser correctly
disabled its Heatmap button. This build uses `typing.Optional` and
`typing.Tuple` in `detector/heatmap.py` for Python 3.9-compatible imports.
Re-run a scan after deployment; an old result stored in browser sessionStorage
will not gain a missing heatmap automatically.
