from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import from_dydema_prediction
from .evaluate import StreamEvent, calibrate_threshold, evaluate_stream, synthetic_demo
from .gate import GateConfig, IdleGate


def read_events(path: str) -> list[StreamEvent]:
    events = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        try:
            truth = str(item.get("truth", "rest"))
            probs = item.get("probabilities") or from_dydema_prediction(item)["probabilities"]
            events.append(StreamEvent(truth, {str(k): float(v) for k, v in probs.items()}, int(item.get("time_ms", line_no * 250))))
        except Exception as exc:
            raise ValueError(f"line {line_no}: {exc}") from exc
    return events


def cmd_simulate(args) -> int:
    report = evaluate_stream(synthetic_demo(args.step_ms), GateConfig())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_evaluate(args) -> int:
    report = evaluate_stream(read_events(args.input), GateConfig())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_calibrate(args) -> int:
    result = calibrate_threshold(read_events(args.input), args.target_false_per_minute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["feasible"] else 2


def cmd_gate(args) -> int:
    gate = IdleGate(GateConfig())
    rows = []
    for event in read_events(args.input):
        d = gate.update({"probabilities": event.probabilities}, now_ms=event.time_ms)
        rows.append({"time_ms": event.time_ms, "state": d.state, "command": d.command, "fired": d.fired, "probability": d.probability, "margin": d.margin, "drift": d.drift, "reason": d.reason, "audit_hash": d.audit_hash})
    print(json.dumps({"decisions": rows, "final_audit_hash": gate.audit_hash}, indent=2, sort_keys=True))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="idle-gate", description="Abstain before a brain-command probability becomes an action.")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--step-ms", type=int, default=250)
    s.set_defaults(func=cmd_simulate)
    s = sub.add_parser("evaluate")
    s.add_argument("input")
    s.set_defaults(func=cmd_evaluate)
    s = sub.add_parser("calibrate")
    s.add_argument("input")
    s.add_argument("--target-false-per-minute", type=float, default=0.2)
    s.set_defaults(func=cmd_calibrate)
    s = sub.add_parser("gate")
    s.add_argument("input")
    s.set_defaults(func=cmd_gate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
