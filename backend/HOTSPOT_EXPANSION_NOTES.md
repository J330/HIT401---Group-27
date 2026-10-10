# Localized hotter-red heatmap update

This version expands existing high-attribution hot areas by a small radius (~1.8% of the image
short side; capped at 18 px) and slightly amplifies their warmth. The operation occurs in
`detector/heatmap.py` after Grad-CAM generation and before the web/PDF overlays are created,
so it affects both views. It does not change predictions, model weights, or confidence.

The operation does NOT detect Sigatoka lesions: it expands any region the classifier already
focuses on, but the current foreground-mask update now clips expansion to the detected leaf. Red expanded pixels are display-only,
not direct attribution measurements. The map cannot identify lesion boundaries. For true
lesion-localized overlays, train/evaluate a separate segmentation model on annotated spots
or evaluate attribution localization against independently annotated lesion masks.

Variables in `detector/heatmap.py`:
- `HOTSPOT_SEED_THRESHOLD`: default 0.74; the minimum display-normalized attention that seeds growth.
- `HOTSPOT_RADIUS_FRACTION`: default 0.018; how far the highlight grows (relative to image size).
- `HOTSPOT_STRENGTH`: default 1.16; how warm the nearby pixels become.

Run tests: `python -m unittest discover -s tests -v`
