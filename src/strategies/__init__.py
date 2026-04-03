"""Strategy and risk overlays built on top of the core/engine split.

The project keeps data handling, signal generation, execution, and risk
management separate so each layer stays small and testable. Strategy classes
focus on *what* the portfolio wants to do; risk managers focus on *how much* to
trade; the engine owns scheduling and execution bookkeeping.

References:
[1] Lopez de Prado, *Advances in Financial Machine Learning*.
[2] Chan, *Algorithmic Trading*.
"""

from .baseline_noise_area import BaselineNoiseAreaStrategy
from .ml_overlay_sizer import MLOutputSizerRiskManager

__all__ = [
    "BaselineNoiseAreaStrategy",
    "MLOutputSizerRiskManager",
]
