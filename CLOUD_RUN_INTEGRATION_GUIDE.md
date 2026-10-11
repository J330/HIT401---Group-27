# HIT401 – Cloud Run deployment (merged from Django(4).zip)

## What was actually transferred

From the older **Django(4).zip**: a compact Python 3.11 Docker image, `libgomp1`,
CPU-only PyTorch, `collectstatic` + WhiteNoise, a single Gunicorn worker bound to
Cloud Run's `${PORT}`, and allowing Django to start without demanding all model
weights at import time. The newer site's Django version, urls, camera,
PDF, frontend, classifier architecture, and heatmap implementation are retained.

**Important:** The old ZIP has no model weights (`trainedai/` is absent). A site
that loads successfully does not prove its AI detection works. For this branch,
you still need real `ml/models/gate_centroids.pt` and
`ml/models/convnextv2base.pt` (or another classifier's checkpoint). These are
**different weights** from the old five-disease DINOv2 project. Never substitute
old DINOv2 `.pth` files for these `.pt` files.

## Folder layout

Keep this at the ROOT of your GitHub repository:

```text
HIT401---Group-27/
  Dockerfile
  cloudbuild.yaml
  .dockerignore
  .gcloudignore
  deploy-cloud-run.ps1
  cloudrun/
  backend/
    manage.py
    backend/
    detector/
    static/
    templates/
  ml/
    models/
      gate_centroids.pt          # real weights, not pointer
      convnextv2base.pt         # real weights, not pointer
```

## Recommended easiest deployment: local source, using real Git LFS files

1. Open PowerShell at your **repository root** and run:

```powershell
git lfs install
git lfs pull
Get-Item .\ml\models\*.pt | Select-Object Name,Length
python cloudrun/check_models.py --model convnextv2
```

2. Authenticate/configure your Google Cloud CLI and project; ensure the
`django-secret` Secret Manager secret exists and the Cloud Run runtime service
identity can read it. Cloud Run/Cloud Build permissions and the deployment
APIs must be set up too.

3. Run:

```powershell
.\deploy-cloud-run.ps1 -ProjectId YOUR_PROJECT_ID
```

This uses **`gcloud run deploy --source .`**, so the build context contains the
actual weights from your **local** filesystem, rather than GitHub's LFS pointers.
The direct local workflow does **not** require a `cloudbuild.yaml` trigger or a
Cloud Storage bucket. It still uses Cloud Build under the hood to build the
Docker image, and incurs ordinary Cloud charges.

If `git lfs pull` fails (quota/permissions/weights not on GitHub LFS), place
the original trained `.pt` files inside `ml/models/`, or pass an existing
Google Cloud Storage bucket to the PowerShell helper:

```powershell
.\deploy-cloud-run.ps1 -ProjectId YOUR_PROJECT_ID -ModelBucket YOUR_BUCKET
```

## GitHub auto-deployment: Cloud Build trigger

A GitHub-linked Cloud Build trigger receives **LFS pointers**, not trained
checkpoint bytes. `cloudbuild.yaml` intentionally checks this and fails early;
this is correct to prevent deploying an AI detector that cannot scan.

- Upload the real `gate_centroids.pt` and `convnextv2base.pt` to the root of a
  Cloud Storage bucket. The Cloud Build service account needs
  `roles/storage.objectViewer` for that bucket.
- Set `_MODEL_BUCKET` to the bucket's name (no `gs://`) in `cloudbuild.yaml`
  **and/or** in the Cloud Build trigger substitutions (the trigger takes priority).
- Re-run the trigger. Step 0 downloads the model files, Step 1 verifies them,
  then Docker builds and Cloud Run deploys.

## Website-only caveat

Like the older Dockerfile, the revised root Dockerfile can build/start without
model checkpoints. This is useful for debugging the webpage, but the scan API
returns HTTP 503 until it can load the real files. The PowerShell deploy helper
and GitHub Cloud Build trigger prevent this by explicitly checking models.

The Cloud Run service runs with one worker, HTTPS proxy support, WhiteNoise,
static assets, model backbone caching and ephemeral storage. The PDF report is
browser-side and will be printable from your Cloud Run HTTPS domain. The
camera requires browser permission. Min instances is 0; billing can still apply.

## Verification

* Open the deployed HTTPS URL. Check Home, Upload, Result layout and CSS.
* Upload a known test leaf. Check that both DINOv2 gate and selected classifier
  return an actual prediction (not HTTP 503). Confirm Heatmap / Original toggle.
* Test Save as PDF and ensure it stays one A4 page.
* If scan fails, read Cloud Run logs; compare active model and checkpoint names.

## Difference between the two deployments

**Older:** `Django/Dockerfile` builds the web server without an explicit trained
weights check. It doesn't include a Cloud Build trigger YAML in the ZIP.
**Current:** `Dockerfile` uses that startup style, adapted to `backend/` plus
`ml/`; `cloudbuild.yaml` still enforces trained checkpoints for a real scan.
