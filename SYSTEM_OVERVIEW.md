# System Overview

This document explains the project in plain order:

1. how the idea started,
2. what the first paper-style results looked like,
3. what realism means in this repository,
4. how realism changed the conclusions,
5. which realistic variants survived,
6. how the live websocket system is separated from the research metrics,
7. and how the whole project was containerized and deployed.

Repository:

- [intraday-momentum-ml](https://github.com/uvgusakk/intraday-momentum-ml)

## 1. What This Project Is

This repository studies a single-instrument intraday momentum strategy for SPY.

The work is intentionally split into two layers:

1. a research and backtesting layer
2. a live websocket and paper-trading layer

That separation matters. The live runtime is not allowed to overwrite or
reinterpret the validated research metrics.

## 2. How The Project Started

The starting point was a paper-style baseline strategy inspired by the SPY
intraday momentum / Noise Area breakout structure.

Core baseline logic:

- build a 14-day minute-of-day volatility profile,
- construct gap-adjusted breakout bands `UB` and `LB`,
- make decisions on a fixed intraday cadence,
- enter in the breakout direction,
- stop against the tighter of the current band and VWAP,
- flatten by the end of the day.

The original goal was simple:

- reproduce a coherent baseline,
- then see whether an ML overlay could improve sizing without replacing the
  baseline signal itself.

Relevant code:

- [baseline_strategy.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/baseline_strategy.py)
- [indicators.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/indicators.py)
- [preprocess.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/preprocess.py)

## 3. The First Extension: ML As A Meta-Model

The ML layer was not designed to predict the whole market from scratch.

Instead, it acts as a meta-model:

- the baseline still proposes candidate entries,
- the model scores those baseline opportunities,
- and capital is scaled up or down based on estimated trade quality.

This design is easier to defend than a fully standalone intraday classifier,
because the baseline already filters the market down to structured event windows.

Relevant code:

- [features_ml.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/features_ml.py)
- [train_ml.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/train_ml.py)
- [backtest_ml_filter.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/backtest_ml_filter.py)

## 4. First Paper-Style Result Before Realism

Before realistic execution frictions were added, the score-forward comparison
looked materially stronger.

Paper-style pre-realism summary:

| method | total_return_pct | sharpe | max_drawdown |
|---|---:|---:|---:|
| `soft` | `42.574393` | `0.915792` | `-0.124481` |
| `mm` | `34.383715` | `0.844447` | `-0.129612` |
| `baseline` | `32.748903` | `0.748184` | `-0.131480` |

Why those numbers were not enough:

- fills were too optimistic,
- stop handling was too forgiving,
- and the resulting Sharpe could not be treated as execution-realistic.

So this stage is part of the research story, but not the final standard the
project now trusts.

## 5. What Realism Means In This Repository

Realism here does not mean a full exchange simulator. It means correcting the
largest backtest assumptions that were likely flattering the strategy.

In this repository, realism means:

1. **next-bar execution**
   - do not decide on a bar close and fill on that same close
   - instead, act on the next available bar
2. **minute-by-minute stop monitoring**
   - do not wait only for the next 30-minute decision checkpoint
   - monitor stop conditions on the minute bars
3. **spread/slippage proxy**
   - include a simple adverse execution penalty instead of only a flat
     commission model

These changes are important because they affect exactly the parts that can make
an intraday backtest look unrealistically smooth:

- entries,
- exits,
- churn,
- drawdown,
- and the Sharpe ratio.

## 6. Why Realism Became Necessary

The early results looked stronger than they should have.

The main realism problems were:

1. same-bar close execution was too optimistic,
2. stops were only checked at decision times,
3. the backtest ignored a basic spread/slippage layer.

So the project was extended with a more realistic execution model:

- next-bar execution instead of same-bar fills,
- minute-by-minute stop monitoring,
- spread proxy,
- and later, a hybrid stop design to reduce pure stop-out churn.

This changed the conclusions materially. It reduced the apparent quality of the
earlier winner and forced the search to be repeated under realistic assumptions.

Relevant code:

- [backtest_ml_filter.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/backtest_ml_filter.py)
- [scoreforward_eval.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/scoreforward_eval.py)
- [ml_overlay_robust.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/ml_overlay_robust.py)

## 7. The Core Workflow

The workflow is now:

1. fetch SPY minute bars from Alpaca,
2. preprocess and enrich them,
3. run the baseline,
4. build the ML candidate dataset from baseline opportunities,
5. train rolling models,
6. run score-forward evaluation,
7. compare variants under realistic execution assumptions,
8. keep the research metrics path fixed,
9. test the top realistic variants separately in live websocket mode.

The score-forward evaluation is the main research gate because it keeps the
training and evaluation chronology explicit.

Relevant code:

- [scoreforward_eval.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/scoreforward_eval.py)
- [cli.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/cli.py)

## 8. Canonical Research Result Under Realism

The canonical current realistic score-forward summary is:

| method | total_return_pct | sharpe | max_drawdown |
|---|---:|---:|---:|
| `soft` | `20.396904` | `0.383525` | `-0.202688` |
| `mm` | `-1.393326` | `0.037862` | `-0.162954` |
| `baseline` | `-8.313106` | `-0.171875` | `-0.187587` |

Interpretation:

- `soft` is the best of the canonical serious methods,
- `mm` survives only as a benchmark extension,
- `baseline` remains the reference comparator.

This is the research path the repository keeps as canonical.

The reason it is canonical is not that it has the highest number ever observed.
It is canonical because it is the most stable, disciplined summary of the
current serious path:

- `baseline` as the realistic reference,
- `soft` as the current best serious ML overlay,
- `mm` as the retained benchmark extension.

## 9. Realistic Extension Search And The Top 3 Variants

After realism was added, the project explored hybrid stop variants that reduce
noise exits while keeping catastrophic protection between decision points.

The top 3 realistic extensions found in that bounded search were:

| method | total_return_pct | sharpe | max_drawdown |
|---|---:|---:|---:|
| `soft_hybrid_7_5` | `76.950741` | `0.909068` | `-0.221610` |
| `soft_hybrid_10` | `72.510460` | `0.870794` | `-0.255491` |
| `soft_hybrid_5` | `62.298459` | `0.801717` | `-0.198020` |

What each of these does:

- `soft_hybrid_7_5`
  - keeps the realistic soft-size overlay,
  - but treats minute stop breaches as immediately actionable only when they are
    catastrophic by 7.5 bps
  - goal: cut noisy stop-outs without removing emergency protection
- `soft_hybrid_10`
  - same idea, but with a looser 10 bps catastrophic threshold
  - this reduces churn further, but allows somewhat larger adverse excursions
- `soft_hybrid_5`
  - same idea, but with a tighter 5 bps catastrophic threshold
  - this is the most conservative of the top 3 and has the best drawdown of the
    three hybrid winners

Why these extensions were tested:

- once realistic minute-stop monitoring was added, pure stop churn became one of
  the main performance drags
- the hybrid stop mechanism was designed to preserve catastrophic protection
  while avoiding needless exit/re-entry loops on smaller intraminute touches

These are important, but they are deliberately kept separate from the canonical
`soft/mm/baseline` research summary. The reason is discipline:

- the canonical path should stay stable,
- the realistic hybrid family should be treated as an extension layer,
- and the live runtime should monitor the strongest realistic candidates without
  rewriting the base research narrative.

This is why the live websocket runtime watches:

- `baseline`
- `soft_hybrid_7_5`
- `soft_hybrid_10`
- `soft_hybrid_5`

Notebook references:

- [03_realistic_soft_improvements.ipynb](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/notebooks/03_realistic_soft_improvements.ipynb)
- [04_unseen_2025_holdout_check.ipynb](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/notebooks/04_unseen_2025_holdout_check.ipynb)

## 10. Live Websocket And Paper-Trading Setup

The live layer is separate from the research layer.

It does not recompute score-forward metrics. Instead, it does this:

1. warm up recent history from Alpaca,
2. connect to the live Alpaca websocket,
3. rebuild the current state on each fresh minute bar,
4. compute the current strategy state for the selected variants,
5. optionally route one chosen variant to Alpaca paper trading.

Important implementation details:

- warmup uses recent historical bars so indicators are defined before streaming
- the first seeded historical bar is treated as `warming_live_stream`, not as a
  real action bar
- live routing is paper-only unless explicitly changed in code

Live modules:

- [live_alpaca.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/live_alpaca.py)
- [live_strategy_runtime.py](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/src/live_strategy_runtime.py)

CLI entrypoints:

- `python -m src.cli live_strategy_board`
- `python -m src.cli live_paper_strategy`

## 11. Environment Setup

This project expects a local `.env` file, but `.env` itself must never be
committed.

Use:

- [.env.example](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/.env.example)

Workflow:

1. copy `.env.example` to `.env`
2. fill only your real Alpaca credentials locally
3. keep `.env` untracked
4. commit only `.env.example`

Key variables:

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
- `ALPACA_DATA_URL=https://data.alpaca.markets`
- `ALPACA_LIVE_FEED=iex`
- `DATA_DIR=./data`

For this documentation pass, `.env` was not inspected. The committed contract is
the example file only.

## 12. Containerization

The project was then containerized so it could run reproducibly outside the
local notebook environment.

Files:

- [Dockerfile](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/Dockerfile)
- [.dockerignore](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/.dockerignore)
- [run_serious_batch.sh](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docker/run_serious_batch.sh)
- [run_live_strategy.sh](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docker/run_live_strategy.sh)

The batch container and the live websocket container use the same image base,
but different commands.

That separation is deliberate:

- batch research runs are finite,
- live websocket runs are operational monitoring / paper-trading sessions.

## 13. Google Cloud Deployment

The project was deployed to Google Cloud in two separate ways:

1. batch research pipeline as a Cloud Run Job
2. live websocket demo as a separate Cloud Run Job

Why jobs instead of services:

- the batch path is finite by design,
- the live demo was also run for a fixed bounded duration,
- neither path required a public HTTP server.

Deployment helpers:

- [create_secrets_from_env.sh](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/gcp/create_secrets_from_env.sh)
- [deploy_cloud_run_job.sh](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/gcp/deploy_cloud_run_job.sh)

Deployment proof:

- [CLOUD_RUN_PROOF.md](https://github.com/uvgusakk/intraday-momentum-ml/blob/main/docs/CLOUD_RUN_PROOF.md)

## 14. What We End Up With

At the end, the repository contains:

1. a reproducible SPY intraday momentum research pipeline,
2. a documented paper-style pre-realism reference stage,
3. a realistic score-forward evaluation path,
4. a documented set of stronger realistic hybrid candidates,
5. a live Alpaca websocket monitoring and paper-routing layer,
6. a Dockerized runtime,
7. and Google Cloud deployment proof for both batch and live jobs.

The important engineering choice is that these layers are connected, but not
mixed together:

- research metrics stay stable,
- realistic extensions are documented explicitly,
- live trading code is additive,
- deployment proof is versioned.
