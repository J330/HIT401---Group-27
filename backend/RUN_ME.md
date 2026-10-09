# Django + Original Frontend — One Server

Open PowerShell in this backend folder (where manage.py is located):

```powershell
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Visit http://127.0.0.1:8000/ (Home), or http://127.0.0.1:8000/upload.html (Upload).

No `python -m http.server 8080` is needed. All four original frontend pages and their CSS/JS are now hosted by Django. The scan sends POST /api/predict/ on the same origin.

Trained AI checkpoints are NOT in this ZIP. The prediction backend expects files in `../ml/models/` relative to the backend directory. Install the Python dependencies and supply trained files for real scans.

For production, run collectstatic and configure your static-file server as usual.

## Grad-CAM: fixed models and troubleshooting

The Original / Heatmap buttons appear at the top-left of the leaf image.
Heatmap becomes clickable only after a successful Grad-CAM response.
The heatmap explains the *active binary health classifier* (not the DINOv2
open-set gate), and is produced for accepted scans.

The current backend supports the trained classifier keys:
`convnextv2` (default), `efficientnetv2s`, `swin`, and `vit`.

Grad-CAM is implemented in `detector/cam.py` with model-specific layers:
- ConvNeXtV2: last ConvNeXtV2 block (4D spatial activations).
- EfficientNetV2-S: convolution head.
- Swin: final block normalization before attention (spatial tokens).
- ViT: final transformer block normalization *before* attention (patch tokens
  excluding the CLS token).

A previous implementation passed Hugging Face `ImageClassifierOutput` to a
Grad-CAM function expecting raw logits, and selected wrong/unsupported layers.
It also caught and discarded all errors silently. The new code generates
Grad-CAM from real class logits and logs full exceptions to your Django console.
When `DEBUG=True`, an error detail is shown below the image, and the network
response contains `heatmap_error`. In production, only a generic message is
returned. An actual heatmap is never fabricated on failure.

**Project folder placement** (keep your already-trained files):

```text
HIT401---Group-27/
  backend/
    manage.py
    detector/cam.py
    ...
  ml/
    models/
      gate_centroids.pt
      convnextv2base.pt        # used when ACTIVE_MODEL=convnextv2
      effnetv2s.pt            # used when ACTIVE_MODEL=efficientnetv2s
      swinbase.pt             # used when ACTIVE_MODEL=swin
      vitbase.pt              # used when ACTIVE_MODEL=vit
```

Train/checkpoint weights are NOT included in this download. You only need the
file for the active classifier and `gate_centroids.pt` to perform a scan.
The original data and model files can remain unchanged.

Run `python -m unittest discover -s tests -p 'test_cam_smoke.py' -v`
from `backend` to run 6 synthetic unit tests of the CAM logic. These do not
replace checking with the real model checkpoint.

If a real scan still does not show a heatmap, copy the console traceback or
`heatmap_error` value from the POST `/api/predict/` response.
