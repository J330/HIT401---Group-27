#!/usr/bin/env bash
# Cloud Build checks out Git LFS *pointers*, not the model bytes. This script
# fetches only the two required files from the exact GitHub commit.
set -Eeuo pipefail

MODEL_BUCKET="${1:-}"
ACTIVE_MODEL="${2:-convnextv2}"
COMMIT="${3:-}"
REPO_URL="${4:-https://github.com/J330/HIT401---Group-27.git}"

if [[ -n "$MODEL_BUCKET" ]]; then
  echo "Using checkpoints fetched from gs://${MODEL_BUCKET}; Git LFS step skipped."
  exit 0
fi

case "$ACTIVE_MODEL" in
  convnextv2) CLASSIFIER=convnextv2base.pt ;;
  efficientnetv2s) CLASSIFIER=effnetv2s.pt ;;
  swin) CLASSIFIER=swinbase.pt ;;
  vit) CLASSIFIER=vitbase.pt ;;
  *) echo "ERROR: Invalid active model: $ACTIVE_MODEL" >&2; exit 2 ;;
esac

if [[ -z "$COMMIT" || "$COMMIT" == "unknown" ]]; then
  echo 'ERROR: No COMMIT_SHA available. Set _MODEL_BUCKET to a GCS bucket containing real checkpoints.' >&2
  exit 2
fi

is_real_weights() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  [[ $(wc -c < "$file") -ge 1024 ]] || return 1
  ! head -n 1 "$file" | grep -q '^version https://git-lfs.github.com/spec/v1'
}

if is_real_weights "ml/models/gate_centroids.pt" && is_real_weights "ml/models/$CLASSIFIER"; then
  echo 'Real trained checkpoints are already present; Git LFS download skipped.'
  exit 0
fi

echo "Downloading $CLASSIFIER and gate_centroids.pt from GitHub LFS for commit $COMMIT"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends git git-lfs ca-certificates >/dev/null

# Do not download every trained checkpoint; the model files can be very large.
# Explicitly pull only the gate and the selected health classifier.
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
export GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1

git init -q "$workdir"
git -C "$workdir" remote add origin "$REPO_URL"
if ! git -C "$workdir" -c protocol.version=2 fetch -q --depth=1 origin "$COMMIT"; then
  echo 'ERROR: Could not fetch the GitHub commit (check repository visibility and commit SHA).' >&2
  exit 1
fi
git -C "$workdir" -c advice.detachedHead=false checkout -q FETCH_HEAD
git -C "$workdir" lfs install --local >/dev/null
if ! git -C "$workdir" lfs pull \
    --include="ml/models/gate_centroids.pt,ml/models/$CLASSIFIER" --exclude=""; then
  echo 'ERROR: GitHub LFS pull failed (object missing, quota, or access). Upload weights to GCS and set _MODEL_BUCKET instead.' >&2
  exit 1
fi

mkdir -p ml/models
for name in gate_centroids.pt "$CLASSIFIER"; do
  src="$workdir/ml/models/$name"
  if ! is_real_weights "$src"; then
    echo "ERROR: $name is absent or still a pointer in GitHub LFS. Set _MODEL_BUCKET to use GCS weights." >&2
    exit 1
  fi
  cp "$src" "ml/models/$name"
  echo "Restored ml/models/$name ($(wc -c < "ml/models/$name") bytes)."
done

echo 'Both model files are ready for the normal Cloud Build validation step.'
