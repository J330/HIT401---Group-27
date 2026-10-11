# Run in PowerShell from the REPO ROOT. Requires gcloud CLI and a secret.
param(
    [string]$ProjectId = "",
    [string]$Region = "australia-southeast1",
    [string]$Service = "banana-leaf-detector"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
}
if (-not $ProjectId) { throw "Specify -ProjectId YOUR_GOOGLE_CLOUD_PROJECT_ID or set gcloud config set project YOUR_PROJECT_ID" }
Write-Host "Deploying $Service to $Region in $ProjectId. Cloud Run must have SECRET_KEY secret configured."
python cloudrun/check_models.py --model convnextv2
if ($LASTEXITCODE -ne 0) { throw "Missing model files. See CLOUD_RUN_README.md." }
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
    --set-env-vars "ACTIVE_MODEL=convnextv2,DEBUG=False,SAVE_PREDICTIONS=False" `
    --set-secrets "SECRET_KEY=django-secret:latest" `
    --allow-unauthenticated
if ($LASTEXITCODE -ne 0) { throw "Cloud Run deploy failed. Check the error above." }
