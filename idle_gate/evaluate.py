from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .gate import GateConfig, IdleGate


@dataclass(frozen=True)
class StreamEvent:
    truth: str
    probabilities: dict[str, float]
    time_ms: int


def normalize_prediction(item: Mapping[str, Any]) -> dict[str, float]:
    probs = item.get("probabilities")
    if not isinstance(probs, Mapping):
        raise ValueError("prediction item must have probabilities")
    return {str(k): float(v) for k, v in probs.items()}


def evaluate_stream(events: Iterable[StreamEvent], config: GateConfig | None = None) -> dict[str, Any]:
    gate = IdleGate(config)
    decisions = []
    fires = []
    correct = 0
    total = 0
    first_fire_latency: list[int] = []
    current_command: str | None = None
    command_start = 0
    for event in events:
        d = gate.update({"probabilities": event.probabilities}, now_ms=event.time_ms)
        decisions.append({"time_ms": event.time_ms, "truth": event.truth, "state": d.state, "command": d.command, "reason": d.reason, "drift": d.drift})
        if event.truth != "rest":
            total += 1
            if event.truth != current_command:
                current_command = event.truth
                command_start = event.time_ms
        else:
            current_command = None
        if d.fired:
            fires.append({"time_ms": event.time_ms, "truth": event.truth, "command": d.command})
            if event.truth == d.command:
                correct += 1
                first_fire_latency.append(event.time_ms - command_start)
    false_fires = [f for f in fires if f["truth"] == "rest" or f["truth"] != f["command"]]
    duration_min = 0.0
    if decisions:
        duration_min = max(1e-9, (decisions[-1]["time_ms"] - decisions[0]["time_ms"] + 1) / 60000)
    return {
        "events": len(decisions),
        "fires": len(fires),
        "correct_fires": correct,
        "false_fires": len(false_fires),
        "false_activations_per_minute": len(false_fires) / duration_min,
        "command_fire_rate": correct / total if total else None,
        "mean_fire_latency_ms": (sum(first_fire_latency) / len(first_fire_latency)) if first_fire_latency else None,
        "blocked_windows": sum(1 for d in decisions if d["state"] == "blocked_drift"),
        "final_audit_hash": gate.audit_hash,
        "decisions": decisions,
    }


def calibrate_threshold(events: list[StreamEvent], target_false_per_minute: float = 0.2) -> dict[str, Any]:
    if target_false_per_minute < 0:
        raise ValueError("target_false_per_minute must be non-negative")
    candidates = []
    for enter, margin, dwell in itertools.product([0.65, 0.7, 0.75, 0.8, 0.85, 0.9], [0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]):
        cfg = GateConfig(enter_probability=enter, exit_probability=max(0.5, enter - 0.15), min_margin=margin, dwell_windows=dwell)
        report = evaluate_stream(events, cfg)
        candidates.append({"enter_probability": enter, "min_margin": margin, "dwell_windows": dwell, "report": {k: v for k, v in report.items() if k != "decisions"}})
    feasible = [c for c in candidates if c["report"]["false_activations_per_minute"] <= target_false_per_minute]
    pool = feasible or candidates
    best = max(pool, key=lambda c: ((c["report"]["command_fire_rate"] or 0), -c["report"]["false_activations_per_minute"], -(c["report"]["mean_fire_latency_ms"] or 10**9)))
    best["feasible"] = best in feasible
    return best


def synthetic_demo(step_ms: int = 250) -> list[StreamEvent]:
    """Deterministic Dydema-response-shaped stream: rest, command, noise, then drift."""
    events: list[StreamEvent] = []
    t = 0
    def add(truth, probs, n):
        nonlocal t
        for _ in range(n):
            events.append(StreamEvent(truth, dict(probs), t))
            t += step_ms
    rest = {"left": 0.31, "right": 0.31, "rest": 0.38}
    left_weak = {"left": 0.62, "right": 0.25, "rest": 0.13}
    left_strong = {"left": 0.84, "right": 0.10, "rest": 0.06}
    right_strong = {"left": 0.08, "right": 0.86, "rest": 0.06}
    drift = {"left": 0.03, "right": 0.94, "rest": 0.03}
    add("rest", rest, 24)
    add("left", left_weak, 2)
    add("left", left_strong, 4)
    add("rest", rest, 8)
    add("right", right_strong, 4)
    add("rest", rest, 8)
    add("right", drift, 48)
    return events
