from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GateConfig:
    enter_probability: float = 0.78
    exit_probability: float = 0.60
    min_margin: float = 0.22
    dwell_windows: int = 3
    refractory_ms: int = 900
    history_size: int = 240
    drift_window: int = 40
    drift_js_threshold: float = 0.35
    block_on_drift: bool = True
    neutral_labels: tuple[str, ...] = ("rest", "idle", "neutral")

    def validate(self) -> None:
        if not 0.5 <= self.exit_probability <= self.enter_probability <= 1.0:
            raise ValueError("probabilities must satisfy 0.5 <= exit <= enter <= 1")
        if not 0 <= self.min_margin <= 1:
            raise ValueError("min_margin must be in [0, 1]")
        if self.dwell_windows < 1:
            raise ValueError("dwell_windows must be positive")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms must be non-negative")
        if self.history_size < self.drift_window:
            raise ValueError("history_size must be at least drift_window")
        if self.drift_window < 4:
            raise ValueError("drift_window must be at least 4")
        if not 0 < self.drift_js_threshold <= 1:
            raise ValueError("drift_js_threshold must be in (0, 1]")


@dataclass(frozen=True)
class Decision:
    state: str
    command: str | None
    fired: bool
    probability: float
    margin: float
    reason: str
    drift: float
    audit_hash: str


@dataclass
class _Audit:
    last_hash: str = "0" * 64

    def append(self, item: Mapping[str, Any]) -> str:
        body = dict(item)
        body["previous"] = self.last_hash
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        self.last_hash = hashlib.sha256(encoded).hexdigest()
        return self.last_hash


def _top_two(probabilities: Mapping[str, float]) -> tuple[str, float, float]:
    if len(probabilities) < 2:
        raise ValueError("at least two class probabilities are required")
    values: list[tuple[str, float]] = []
    for label, value in probabilities.items():
        if not isinstance(label, str) or not label:
            raise ValueError("labels must be non-empty strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("probabilities must be finite numbers")
        if value < -1e-9 or value > 1 + 1e-9:
            raise ValueError("probabilities must be in [0, 1]")
        values.append((label, float(value)))
    total = sum(v for _, v in values)
    if abs(total - 1.0) > 0.03:
        raise ValueError(f"probabilities must sum to 1 within 0.03, got {total:.6f}")
    values.sort(key=lambda item: item[1], reverse=True)
    return values[0][0], values[0][1], values[0][1] - values[1][1]


def js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    labels = set(left) | set(right)
    def dist(d: Mapping[str, float]) -> dict[str, float]:
        raw = {k: max(0.0, float(d.get(k, 0.0))) for k in labels}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}
    p, q = dist(left), dist(right)
    m = {k: (p[k] + q[k]) / 2 for k in labels}
    def kl(a, b):
        return sum(a[k] * math.log2(a[k] / b[k]) for k in labels if a[k] > 0)
    return (kl(p, m) + kl(q, m)) / 2


class IdleGate:
    """Stateful abstention layer above a Dydema-style label/probability response.

    It fires only after the same non-neutral command clears probability and margin
    thresholds for a dwell period. A fired command enters a refractory period, and
    distribution drift can block firing instead of letting a shifted stream spam actions.
    """

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        self.config.validate()
        self._candidate: str | None = None
        self._candidate_count = 0
        self._armed = True
        self._last_fire_ms: int | None = None
        self._history: deque[tuple[str, float]] = deque(maxlen=self.config.history_size)
        self._baseline: dict[str, float] | None = None
        self._audit = _Audit()

    @property
    def audit_hash(self) -> str:
        return self._audit.last_hash

    def reset(self) -> None:
        self._candidate = None
        self._candidate_count = 0
        self._armed = True
        self._last_fire_ms = None
        self._history.clear()
        self._baseline = None

    def update(self, prediction: Mapping[str, Any], now_ms: int | None = None) -> Decision:
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError("prediction must contain a probabilities object")
        label, probability, margin = _top_two(probabilities)
        self._history.append((label, probability))
        if self._baseline is None and len(self._history) >= self.config.drift_window:
            self._baseline = self._distribution(list(self._history)[-self.config.drift_window:])
        drift = self._drift_score()
        neutral = label.lower() in self.config.neutral_labels
        state = "idle"
        command = None
        fired = False
        reason = "neutral or below threshold"

        if self.config.block_on_drift and self._baseline is not None and drift > self.config.drift_js_threshold:
            self._candidate = None
            self._candidate_count = 0
            state = "blocked_drift"
            reason = f"label distribution drift {drift:.3f} exceeds {self.config.drift_js_threshold:.3f}"
        elif neutral:
            self._candidate = None
            self._candidate_count = 0
            self._armed = True
            reason = "neutral prediction"
        elif probability < self.config.exit_probability:
            self._candidate = None
            self._candidate_count = 0
            reason = "below exit probability"
        elif not self._armed:
            remaining = 0 if self._last_fire_ms is None else max(0, self.config.refractory_ms - (now - self._last_fire_ms))
            if remaining == 0 and probability < self.config.enter_probability:
                self._armed = True
                reason = "refractory complete and evidence released"
            else:
                state = "refractory"
                reason = f"repeat command suppressed for {remaining} ms"
        elif probability >= self.config.enter_probability and margin >= self.config.min_margin:
            if label == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = label
                self._candidate_count = 1
            if self._candidate_count >= self.config.dwell_windows:
                state = "fire"
                command = label
                fired = True
                reason = f"{self._candidate_count} sustained windows cleared evidence thresholds"
                self._armed = False
                self._last_fire_ms = now
                self._candidate = None
                self._candidate_count = 0
            else:
                state = "candidate"
                reason = f"evidence {self._candidate_count}/{self.config.dwell_windows}"
        else:
            self._candidate = None
            self._candidate_count = 0
            reason = "insufficient probability or margin"

        digest = self._audit.append({
            "time_ms": now,
            "label": label,
            "probability": round(probability, 6),
            "margin": round(margin, 6),
            "state": state,
            "command": command,
            "drift": round(drift, 6),
            "reason": reason,
        })
        return Decision(state, command, fired, probability, margin, reason, drift, digest)

    def _distribution(self, rows: list[tuple[str, float]]) -> dict[str, float]:
        counts: dict[str, float] = {}
        for label, _ in rows:
            counts[label] = counts.get(label, 0.0) + 1.0
        return counts

    def _drift_score(self) -> float:
        if self._baseline is None or len(self._history) < self.config.drift_window:
            return 0.0
        current = self._distribution(list(self._history)[-self.config.drift_window:])
        return js_divergence(self._baseline, current)

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "candidate": self._candidate,
            "candidate_count": self._candidate_count,
            "armed": self._armed,
            "last_fire_ms": self._last_fire_ms,
            "history": list(self._history),
            "baseline": self._baseline,
            "audit_hash": self.audit_hash,
        }
