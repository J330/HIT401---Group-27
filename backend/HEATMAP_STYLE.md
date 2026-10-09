# Heatmap palette: focused blue/cyan -> yellow -> red

The user-provided reference screenshot has a colourful hotspot palette with
blue/cyan fringes, yellow transitions, and red centres. This update applies
that visual style to **real class-attribution Grad-CAM values**. Pixels with
low attention are transparent, so the source photo remains visible.

## Source code

- `detector/cam.py`: class-specific Grad-CAM from trained models. ConvNeXtV2
  now uses the penultimate encoder stage (usually a 14x14 grid instead of 7x7)
  to improve localisation without adding artificial spots. Other classifiers
  retain their previous target layers.
- `detector/heatmap.py`: grayscale CAM and coloured API overlay.
- `static/js/result.js`: interactive overlay with matching colours.
- `static/css/style.css`: legend colour bar and bright original image.

## Important limitation

Changing a colour palette cannot turn an unfocused/low-resolution CAM into
a precise lesion map. Red regions show the model's class attribution, not
verified lesions or segmentation. The hotspot locations depend on the model
and image. If there are only broad hotspots, finer lesion-level localisation
requires higher-resolution attribution or a separately validated segmentation
model; it should not be manufactured using colour thresholds.

Run the server from the backend directory:

```powershell
python manage.py runserver
```

Re-scan an image to generate fresh attention data. There is no retraining.

## High-resolution guided view (revision)

- Class-specific Grad-CAM remains the low-resolution spatial prior.
- A *single* extra gradient with respect to the input is extracted from the same
  forward pass and same target logit; Gradient × Input sharpens that prior.
- The 224×224 class-attribution view is **not a lesion detector**. Small hotspots
  only appear where model gradients support them, so an identical match to a
  reference screenshot is not guaranteed.
- We reduced opacity of the red core and preserved cyan/yellow transitions.
- CSS and JavaScript include a cache-busting query value; rescan after updating.
