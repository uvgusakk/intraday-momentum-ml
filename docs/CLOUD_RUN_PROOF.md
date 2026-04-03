# Cloud Run Proof

This document records the deployed Google Cloud evidence that is stored inside
the repository.

Repository:

- [intraday-momentum-ml](https://github.com/uvgusakk/intraday-momentum-ml)
- [System Overview](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/SYSTEM_OVERVIEW.md)

## 1. Batch Research Job

Deployed target:

- project: `project-5d782c2e-0e9a-4cca-878`
- region: `europe-west1`
- job: `intraday-momentum-ml-batch`
- image repo: `europe-west1-docker.pkg.dev/project-5d782c2e-0e9a-4cca-878/intraday-momentum-ml`
- successful image tag: `v2`

Successful execution:

- execution: `intraday-momentum-ml-batch-jjbnl`
- result: completed successfully
- reported runtime: about 6 minutes 24 seconds

Batch configuration:

- `BATCH_MODE=full_pipeline`
- `BATCH_METHODS=soft,mm`
- `BATCH_SYMBOL=SPY`
- `BATCH_START=2021-01-01`
- `BATCH_END=2026-01-01`
- `DATA_DIR=/tmp/data`

Raw proof files:

- [job_describe.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof/job_describe.txt)
- [executions_list.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof/executions_list.txt)
- [execution_describe.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof/execution_describe.txt)
- [job_logs.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof/job_logs.txt)
- [artifact_registry_images.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof/artifact_registry_images.txt)

## 2. Live Websocket Job

Deployed target:

- project: `project-5d782c2e-0e9a-4cca-878`
- region: `europe-west1`
- job: `intraday-momentum-ml-live-job`

Successful execution:

- execution: `intraday-momentum-ml-live-job-48r74`
- result: completed successfully

Live job purpose:

- warm recent history from Alpaca
- connect to the live Alpaca websocket
- evaluate:
  - `baseline`
  - `soft_hybrid_7_5`
  - `soft_hybrid_10`
  - `soft_hybrid_5`
- run for a bounded live session duration

Raw proof files:

- [job_describe.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof_live/job_describe.txt)
- [executions_list.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof_live/executions_list.txt)
- [execution_describe.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof_live/execution_describe.txt)
- [job_logs.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof_live/job_logs.txt)
- [artifact_registry_images.txt](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/cloud_run_proof_live/artifact_registry_images.txt)

## 3. What This Proves

This repository now contains evidence for all of the following:

1. the project was containerized,
2. the image was pushed to Artifact Registry,
3. a Cloud Run batch job was created and executed successfully,
4. a separate Cloud Run live websocket job was created and executed successfully,
5. the raw GCP command outputs and logs were captured into versioned proof
   files.

## 4. Limitation

Both jobs currently use ephemeral container storage for working files. That is
enough to demonstrate deployment and execution, but it is not the same as a
durable production storage design.

The next cloud-hardening step would be:

- writing final outputs to Google Cloud Storage.
