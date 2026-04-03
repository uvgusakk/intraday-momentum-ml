# Extensions

## Cooldown after stop exit

Stops often cluster in choppy regimes. Immediate re-entry after a stop can increase adverse selection and cost drag because the strategy keeps re-engaging the same noisy move.

A short cooldown after stop exits reduces repeated stop-outs while still preserving trend-day exposure, because exits remain unrestricted and the block only applies to new entries for a small number of decision steps.

Cooldown is available but deprecated; flip hysteresis is the preferred churn control.

## Upcoming structural filters

The next extensions under consideration are:
- break-strength filter
- flip hysteresis

The intent is to reduce churn without relying on fixed time-of-day constraints that underperformed in testing.

## Robust ML Soft Sizing Overlay

The robust soft sizing overlay keeps the baseline direction logic unchanged and only adjusts position size. It is designed to reduce drawdown and improve Sharpe by making the ML overlay less sensitive to unstable probability scale.

It combines four mechanisms:
- rank-based sizing: use the rolling percentile rank of `p_good` instead of its raw level, which is more stable under model drift and calibration shifts
- neutral-zone mapping: keep size at `1.0` in the noisy middle of the rank distribution and only resize the tails
- shrinkage toward one: damp raw multipliers before execution so small ranking differences do not create large exposure changes
- overlay and risk gating: disable the overlay when recent top-vs-bottom score spread is not positive, and tighten caps in high-risk intraday conditions

In practice, this reduces the chance that a temporarily miscalibrated model or a turbulent regime will cause oversized bets. The expected effect is lower sizing noise, lower drawdown, and a more stable Sharpe profile than direct probability-based scaling.
