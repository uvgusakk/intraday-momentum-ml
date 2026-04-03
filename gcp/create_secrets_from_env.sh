#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"

if [[ ! -f .env ]]; then
  echo ".env not found in $ROOT_DIR" >&2
  exit 1
fi

ALPACA_API_KEY="$(grep '^ALPACA_API_KEY=' .env | cut -d= -f2-)"
ALPACA_API_SECRET="$(grep '^ALPACA_API_SECRET=' .env | cut -d= -f2-)"

if [[ -z "$ALPACA_API_KEY" || -z "$ALPACA_API_SECRET" ]]; then
  echo "ALPACA_API_KEY or ALPACA_API_SECRET missing in .env" >&2
  exit 1
fi

gcloud config set project "$GCP_PROJECT_ID" >/dev/null
gcloud services enable secretmanager.googleapis.com

for secret_name in alpaca-api-key alpaca-api-secret; do
  if ! gcloud secrets describe "$secret_name" >/dev/null 2>&1; then
    gcloud secrets create "$secret_name" --replication-policy automatic
  fi
done

printf '%s' "$ALPACA_API_KEY" | gcloud secrets versions add alpaca-api-key --data-file=-
printf '%s' "$ALPACA_API_SECRET" | gcloud secrets versions add alpaca-api-secret --data-file=-

echo "[gcp] uploaded latest secret versions for alpaca-api-key and alpaca-api-secret"
