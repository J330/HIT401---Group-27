# HIT401 Cloud Build — automatic Git LFS model recovery

This fixes Cloud Build stopping with `gate_centroids.pt is a Git LFS pointer`.

## How it works

1. `Fetch model checkpoints` downloads weights from Cloud Storage **if** `_MODEL_BUCKET` is set.
2. `Resolve GitHub LFS checkpoints` retrieves the two real files directly from the public GitHub repository at `$COMMIT_SHA` **if** `_MODEL_BUCKET` is empty. It needs a working GitHub LFS object and available Git LFS bandwidth. Only `gate_centroids.pt` and the currently selected classifier are fetched.
3. `Verify model checkpoints` rejects missing weights, text pointers, or implausibly small files.
4. Docker build, image push, and Cloud Run deployment proceed *only if* those checkpoints are valid.

## Install (GitHub trigger)

Copy both files into your repository, preserving the second file's folder:

```
HIT401---Group-27/
    cloudbuild.yaml                     (replace existing)
    cloudrun/
        fetch_git_lfs.sh               (new)
        check_models.py                (existing)
```

From the repository root in PowerShell:

```powershell
git add cloudbuild.yaml cloudrun/fetch_git_lfs.sh
git commit -m "Fetch Git LFS model weights in Cloud Build"
git push
```

Ensure your Cloud Build trigger points to `cloudbuild.yaml` at the root. The repository URL is configured under `_GIT_LFS_REPO` in that file.

## If the new Git LFS step fails

GitHub LFS can fail because trained objects were never pushed, LFS bandwidth is exceeded, or downloading anonymously is restricted. In that case, use the Cloud Storage fallback:

- Run `git lfs pull` in your local repository and ensure the two `.pt` files are real, not tiny text pointers.
- Upload `gate_centroids.pt` and the selected classifier `.pt` to the **root** of your Cloud Storage bucket.
- Set `_MODEL_BUCKET: 'YOUR_BUCKET_NAME'` (without `gs://`) in `cloudbuild.yaml` or in the trigger's substitutions. The Cloud Build service account needs Storage Object Viewer access to that bucket.
- Rerun the trigger.

Or avoid the GitHub trigger and deploy from your local source with real weights:

```powershell
git lfs pull
python cloudrun/check_models.py --model convnextv2
.\deploy-cloud-run.ps1 -ProjectId YOUR_PROJECT_ID
```

The latter still uses Cloud Build for the container build but uploads your local files instead of GitHub's pointer-only checkout.

## Notes

- Do not delete `Verify model checkpoints`: removing it can deploy a site where scans fail.
- This patch does not alter the Django frontend, prediction algorithms, trained weights, or PDF/heatmap code.
- The actual Google Cloud build, GitHub LFS download, and Cloud Run deployment have **not** been run here; syntax and simulated-flow tests passed.
