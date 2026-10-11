# Cloud Run integration changes

- Adapted Dockerfile setup from `Django(4).zip` to newer `backend/` + `ml/` structure.
- Added `libgomp1` and port-aware one-worker Gunicorn startup, CPU PyTorch, static collection/WhiteNoise.
- Deferred **both** classifier and DINOv2 gate loading until a prediction is requested.
- If an unconfigured website receives a scan, it now responds with HTTP 503 and logs the model error instead of crashing Django at import time.
- Kept newer heatmap, PDF, camera, API, templates, Django settings and all model choices.
- Cloud Build trigger still requires valid checkpoints; `_MODEL_BUCKET` can download from GCS.
- PowerShell helper now supports alternate classifier models and optional GCS downloading; local source deploy avoids GitHub's Git LFS pointer checkout once `git lfs pull` succeeded.

The old ZIP had no trained model files. This merge does not claim it can generate AI predictions without the correct trained weights.
