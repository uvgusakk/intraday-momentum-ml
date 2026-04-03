#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:?Set GCP_REGION, e.g. europe-west1}"

REPOSITORY="${GCP_ARTIFACT_REPO:-intraday-momentum-ml}"
IMAGE_NAME="${GCP_IMAGE_NAME:-intraday-momentum-ml}"
IMAGE_TAG="${GCP_IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
JOB_NAME="${GCP_JOB_NAME:-intraday-momentum-ml-batch}"
SERVICE_ACCOUNT="${GCP_SERVICE_ACCOUNT:-}"

SECRET_KEY_NAME="${GCP_SECRET_ALPACA_KEY_NAME:-alpaca-api-key}"
SECRET_SECRET_NAME="${GCP_SECRET_ALPACA_SECRET_NAME:-alpaca-api-secret}"

BATCH_MODE="${BATCH_MODE:-scoreforward_only}"
BATCH_METHODS="${BATCH_METHODS:-soft,mm}"
BATCH_SYMBOL="${BATCH_SYMBOL:-SPY}"
BATCH_START="${BATCH_START:-2021-01-01}"
BATCH_END="${BATCH_END:-2026-01-01}"
BATCH_LABEL_MODE="${BATCH_LABEL_MODE:-baseline_trade}"
BATCH_HORIZON_MINS="${BATCH_HORIZON_MINS:-60}"
BATCH_TRAIN_MONTHS="${BATCH_TRAIN_MONTHS:-24}"
BATCH_VAL_MONTHS="${BATCH_VAL_MONTHS:-3}"
BATCH_TEST_MONTHS="${BATCH_TEST_MONTHS:-3}"
BATCH_STEP_MONTHS="${BATCH_STEP_MONTHS:-3}"
BATCH_WARMUP_DAYS="${BATCH_WARMUP_DAYS:-90}"

REPOSITORY_HOST="${GCP_REGION}-docker.pkg.dev"
IMAGE_URI="${REPOSITORY_HOST}/${GCP_PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[gcp] project=$GCP_PROJECT_ID region=$GCP_REGION repo=$REPOSITORY job=$JOB_NAME"
echo "[gcp] image=$IMAGE_URI"

gcloud config set project "$GCP_PROJECT_ID" >/dev/null
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com

if ! gcloud artifacts repositories describe "$REPOSITORY" --location "$GCP_REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format docker \
    --location "$GCP_REGION" \
    --description "Container repo for intraday momentum batch jobs"
fi

gcloud auth configure-docker "${REPOSITORY_HOST}" --quiet

docker build -t intraday-momentum-ml:gcp .
docker tag intraday-momentum-ml:gcp "$IMAGE_URI"
docker push "$IMAGE_URI"

COMMON_ARGS=(
  --image "$IMAGE_URI"
  --region "$GCP_REGION"
  --tasks 1
  --max-retries 0
  --cpu 2
  --memory 4Gi
  --set-env-vars "DATA_DIR=/tmp/data,BATCH_MODE=${BATCH_MODE},BATCH_METHODS=${BATCH_METHODS},BATCH_SYMBOL=${BATCH_SYMBOL},BATCH_START=${BATCH_START},BATCH_END=${BATCH_END},BATCH_LABEL_MODE=${BATCH_LABEL_MODE},BATCH_HORIZON_MINS=${BATCH_HORIZON_MINS},BATCH_TRAIN_MONTHS=${BATCH_TRAIN_MONTHS},BATCH_VAL_MONTHS=${BATCH_VAL_MONTHS},BATCH_TEST_MONTHS=${BATCH_TEST_MONTHS},BATCH_STEP_MONTHS=${BATCH_STEP_MONTHS},BATCH_WARMUP_DAYS=${BATCH_WARMUP_DAYS}"
  --set-secrets "ALPACA_API_KEY=${SECRET_KEY_NAME}:latest,ALPACA_API_SECRET=${SECRET_SECRET_NAME}:latest"
)

if [[ -n "$SERVICE_ACCOUNT" ]]; then
  COMMON_ARGS+=(--service-account "$SERVICE_ACCOUNT")
fi

if gcloud run jobs describe "$JOB_NAME" --region "$GCP_REGION" >/dev/null 2>&1; then
  gcloud run jobs update "$JOB_NAME" "${COMMON_ARGS[@]}"
else
  gcloud run jobs create "$JOB_NAME" "${COMMON_ARGS[@]}"
fi

echo
echo "[gcp] deployed Cloud Run Job: $JOB_NAME"
echo "[gcp] execute with:"
echo "gcloud run jobs execute $JOB_NAME --region $GCP_REGION --wait"
