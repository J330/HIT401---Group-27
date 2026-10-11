# Deploy the fully functional AI site from your LOCAL clone (not a GitHub
# Cloud Build trigger). This method uploads the actual local Git LFS bytes.
param(
    [Parameter(Mandatory=$false)][string]$ProjectId = "",
    [string]$Region = "australia-southeast1",
    [string]$Service = "banana-leaf-detector",
    [ValidateSet("convnextv2","efficientnetv2s","swin","vit")]
    [string]$Model = "convnextv2",
    [string]$ModelBucket = "",
    [string]$DjangoSecret = "django-secret"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
}
if (-not $ProjectId) { throw "Specify -ProjectId YOUR_GOOGLE_CLOUD_PROJECT_ID" }
$weights = @{
    convnextv2="convnextv2base.pt"
    efficientnetv2s="effnetv2s.pt"
    swin="swinbase.pt"
    vit="vitbase.pt"
}
if ($ModelBucket) {
    New-Item -Force -ItemType Directory ml/models | Out-Null
    Write-Host "Downloading real trained weights from gs://$ModelBucket ..."
    gcloud storage cp "gs://$ModelBucket/gate_centroids.pt" "ml/models/gate_centroids.pt"
    if ($LASTEXITCODE -ne 0) { throw "Could not download gate_centroids.pt" }
    gcloud storage cp "gs://$ModelBucket/$($weights[$Model])" "ml/models/$($weights[$Model])"
    if ($LASTEXITCODE -ne 0) { throw "Could not download $($weights[$Model])" }
} else {
    Write-Host "Using real weights from local ml/models/ (run git lfs pull first)."
}
python cloudrun/check_models.py --model $Model
if ($LASTEXITCODE -ne 0) {
    throw "Real trained model files are required for working scans. Run 'git lfs pull' or pass -ModelBucket BUCKET_NAME."
}
Write-Host "Deploying trained $Model detector via root Dockerfile."
# --source . builds from local file bytes, avoiding GitHub's pointer-only checkout.
gcloud run deploy $Service `
    --project $ProjectId `
    --source . `
    --region $Region `
    --memory 8Gi `
    --cpu 2 `
    --concurrency 1 `
    --timeout 900 `
    --min-instances 0 `
    --max-instances 1 `
    --set-env-vars "ACTIVE_MODEL=$Model,DEBUG=False,SAVE_PREDICTIONS=False" `
    --set-secrets "SECRET_KEY=$($DjangoSecret):latest" `
    --allow-unauthenticated
if ($LASTEXITCODE -ne 0) { throw "Cloud Run deploy failed. Check output above." }
