# Cloud Run Proof

This document summarizes the successful Google Cloud deployment evidence that is
stored in the repository.

## Deployed Batch Target

- Project: `project-5d782c2e-0e9a-4cca-878`
- Region: `europe-west1`
- Job: `intraday-momentum-ml-batch`
- Image repo: `europe-west1-docker.pkg.dev/project-5d782c2e-0e9a-4cca-878/intraday-momentum-ml`
- Successful image tag: `v2`

## Successful Execution

- Execution name: `intraday-momentum-ml-batch-jjbnl`
- Result: completed successfully
- Reported runtime: about 6 minutes 24 seconds

The batch job was configured with:

- `BATCH_MODE=full_pipeline`
- `BATCH_METHODS=soft,mm`
- `BATCH_SYMBOL=SPY`
- `BATCH_START=2021-01-01`
- `BATCH_END=2026-01-01`
- `DATA_DIR=/tmp/data`

## Raw Evidence Files

The raw command outputs used as deployment proof are stored here:

- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/job_describe.txt](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/job_describe.txt)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/executions_list.txt](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/executions_list.txt)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/execution_describe.txt](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/execution_describe.txt)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/job_logs.txt](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/job_logs.txt)
- [/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/artifact_registry_images.txt](/Users/ulianahusak/WUTIS_2026/intraday_momentum_ml/docs/cloud_run_proof/artifact_registry_images.txt)

## What This Proves

This repository now contains evidence for all of the following:

1. the project was containerized
2. the image was pushed to Artifact Registry
3. the Cloud Run Job was created
4. the job executed successfully in Google Cloud
5. the execution logs and job metadata were captured into versioned proof files

## Limitation

The current Cloud Run batch job writes working files to `/tmp/data`, which is
ephemeral container storage. That is enough to demonstrate deployment and
execution, but not enough for durable cloud artifact retention. The next cloud
improvement would be writing final outputs to Google Cloud Storage.
