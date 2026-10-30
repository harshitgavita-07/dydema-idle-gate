"""Idle and confidence gating for brain-command classifiers."""

from .gate import GateConfig, IdleGate, Decision
from .evaluate import evaluate_stream, calibrate_threshold

__all__ = ["GateConfig", "IdleGate", "Decision", "evaluate_stream", "calibrate_threshold"]
__version__ = "0.1.0"
