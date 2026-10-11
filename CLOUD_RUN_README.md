# HIT401 Group 27 — Deploy this Django website to Google Cloud Run

**Use the repository root** `HIT401---Group-27/` as the Docker build context. The supplied `Dockerfile` expects sibling directories `backend/` and `ml/models/`, as in your screenshot. Never run `docker build` from `backend/`.

## Files added

- `Dockerfile` — Python 3.11 CPU-only Torch container, preloads pretrained base backbones, collects static files, starts Gunicorn on `$PORT`.
- `.dockerignore`, `.gcloudignore` — keep training datasets, local SQLite DB and Git history out of uploads/builds; keep model weights.
- `backend/requirements-cloudrun.txt` — Django/AI runtime dependencies.
- `cloudrun/check_models.py` — catch absent checkpoints and Git LFS pointers before building.
- `cloudrun/precache_models.py` — cache the DINOv2 gate and selected classifier pretrained backbones at build time.
- `cloudbuild.yaml` — Cloud Build image build, push and Cloud Run deploy for GitHub triggers.
- `deploy-cloud-run.ps1` — optional Windows PowerShell helper for one-off deployment.
- `backend/backend/settings.py`, `backend/detector/views.py` — production host/secret/security settings; no SQLite prediction saves when Cloud Run is stateless.

## 1. Put your two trained checkpoint files in the correct place

```
HIT401---Group-27/
  Dockerfile
  cloudbuild.yaml
  backend/
    manage.py
    backend/settings.py
  ml/
    models/
      gate_centroids.pt
      convnextv2base.pt
```

The files above are **not included** in this ZIP: you must copy them from your training project. Other model checkpoints are optional. Run `python cloudrun/check_models.py --model convnextv2` to check. The source deployment only needs the actual checkpoint files, not the training image datasets.

### GitHub / Git LFS caution

Cloud Build may check out Git LFS POINTERS rather than real weights, depending on your trigger/source configuration. A 130-byte `.pt` text file is not a model. For repeatable CI deployments, upload the two actual files to a private Google Cloud Storage bucket and set `_MODEL_BUCKET` on the Cloud Build trigger; `cloudbuild.yaml` downloads them into the build workspace before the Docker build. The Cloud Build service account needs access to the bucket.

## 2. Initial Google Cloud setup

Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), log in, select your existing project, and enable APIs:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

Generate a secret value locally (PowerShell):

```powershell
python -c "import secrets; from pathlib import Path; Path('django-secret-temp.txt').write_text(secrets.token_urlsafe(64))"
gcloud secrets create django-secret --replication-policy=automatic
gcloud secrets versions add django-secret --data-file=django-secret-temp.txt
Remove-Item django-secret-temp.txt
```

If `django-secret` already exists, skip the create line and add a new version. Grant the **Cloud Run runtime service identity** `Secret Manager Secret Accessor` (`roles/secretmanager.secretAccessor`) on this secret. With Cloud Build deployment, its deployer identity additionally needs Cloud Run Admin, Service Account User, and Artifact Registry permissions; see Google Cloud IAM docs. Never commit a SECRET_KEY or the temp file.

## 3. Simplest way: deploy from your local repository folder

From `HIT401---Group-27/` (not from `backend/`), in PowerShell:

```powershell
python cloudrun/check_models.py --model convnextv2
gcloud run deploy banana-leaf-detector --source . --region australia-southeast1 --memory 8Gi --cpu 2 --concurrency 1 --timeout 900 --min-instances 0 --max-instances 1 --set-env-vars ACTIVE_MODEL=convnextv2,DEBUG=False,SAVE_PREDICTIONS=False --set-secrets SECRET_KEY=django-secret:latest --allow-unauthenticated
```

Or run `./deploy-cloud-run.ps1 -ProjectId YOUR_PROJECT_ID`. The build can take a while: PyTorch and both Hugging Face backbones are large. `--source .` finds the root Dockerfile and uses Cloud Build/Artifact Registry, without needing local Docker. It prints your deployed HTTPS URL. Camera access works over HTTPS after the browser/user grants permission.

If you change the classifier, update the `ACTIVE_MODEL` **Docker build argument** and Cloud Run runtime environment value to match, and provide its `.pt` file. Default is ConvNeXtV2.

## 4. Continuous deployment through `cloudbuild.yaml`

1. Create an Artifact Registry Docker repository named `cloud-run-images` in `australia-southeast1` (or change `_REPOSITORY`).
2. In Google Cloud Console > Cloud Build > Triggers, create a GitHub push trigger pointing at **`cloudbuild.yaml` in repo root**.
3. Set `_REGION`, `_SERVICE_NAME`, `_ACTIVE_MODEL`, `_DJANGO_SECRET` as needed.
4. Prefer `_MODEL_BUCKET=YOUR_EXISTING_BUCKET_NAME` for actual weight storage. Put `gate_centroids.pt` and `convnextv2base.pt` at the bucket root; set IAM on the build identity for GCS read. If models are available as real bytes in the repo checkout, leave `_MODEL_BUCKET` blank.
5. Give the trigger's Cloud Build service account Artifact Registry Writer, Cloud Run Admin, Service Account User (for the Run service identity), and any additional required Secret Manager / GCS access. Google Cloud project policies vary; review least privilege grants.

The build has `options.logging: CLOUD_LOGGING_ONLY` to accommodate builds using an explicit service account. The image embeds the weights and pretrained backbones, so the Cloud Run container does not need network access to Hugging Face while scanning.

## Notes / troubleshooting

- **This project is stateless on Cloud Run**: uploaded photos stay in memory (frontend uses browser session storage); `SAVE_PREDICTIONS=False` disables writing prediction rows to an ephemeral SQLite DB. Database-backed history/admin requires Cloud SQL and Django database configuration. No migration is needed for the public prediction page in stateless mode.
- **No free-tier guarantee**: defaults of 8 GiB RAM and 2 CPUs are safer for two large PyTorch models; usage, image builds, registry storage and model storage can incur charges. Set `--min-instances 0` to permit idle scaling to zero and `--max-instances 1` to cap simultaneous instances, but this does not impose a billing cap.
- If you see `DisallowedHost`, add your custom domain in `ALLOWED_HOSTS` and its HTTPS origin to `CSRF_TRUSTED_ORIGINS`.
- If `collectstatic` fails, verify WhiteNoise and the `/static/` paths.
- If you see `Missing trained checkpoint`, copy your real weights from `ml/models/` or set `_MODEL_BUCKET` for CI. Avoid Git LFS pointers.
- If the page works but prediction errors occur, inspect Cloud Run logs, check memory/timeouts, and ensure the active classifier weights match its base architecture and two classes.
- The Python packages and CPU-only torch may still need compatible-version adjustments if your existing checkpoint training used materially different library versions. A build-time syntax/static check is not a substitute for a test with your real trained weights.
- Open your site at the HTTPS URL printed by Cloud Run, then upload and scan a photo to verify the heatmap. PDFs print through the browser.
