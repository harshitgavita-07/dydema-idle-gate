import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from idle_gate.adapter import batch_from_dydema_response, from_dydema_prediction
from idle_gate.evaluate import StreamEvent, calibrate_threshold, evaluate_stream, synthetic_demo
from idle_gate.gate import GateConfig, IdleGate, js_divergence


def pred(left=0.1, right=0.8, rest=0.1):
    return {"probabilities": {"left": left, "right": right, "rest": rest}}


def strong(label="right", p=0.84):
    other = (1 - p) / 2
    values = {"left": other, "right": other, "rest": other}
    values[label] = p
    return {"probabilities": values}


def rest():
    return {"probabilities": {"left": 0.31, "right": 0.31, "rest": 0.38}}


class ConfigTests(unittest.TestCase):
    def test_default_config_valid(self):
        GateConfig().validate()

    def test_enter_below_exit_rejected(self):
        with self.assertRaises(ValueError): GateConfig(enter_probability=0.5, exit_probability=0.6).validate()

    def test_exit_below_half_rejected(self):
        with self.assertRaises(ValueError): GateConfig(enter_probability=0.6, exit_probability=0.49).validate()

    def test_margin_negative_rejected(self):
        with self.assertRaises(ValueError): GateConfig(min_margin=-0.1).validate()

    def test_dwell_zero_rejected(self):
        with self.assertRaises(ValueError): GateConfig(dwell_windows=0).validate()

    def test_refractory_negative_rejected(self):
        with self.assertRaises(ValueError): GateConfig(refractory_ms=-1).validate()

    def test_drift_window_too_small_rejected(self):
        with self.assertRaises(ValueError): GateConfig(drift_window=3).validate()

    def test_history_smaller_than_drift_rejected(self):
        with self.assertRaises(ValueError): GateConfig(history_size=10, drift_window=20).validate()

    def test_drift_threshold_bounds(self):
        with self.assertRaises(ValueError): GateConfig(drift_js_threshold=0).validate()
        with self.assertRaises(ValueError): GateConfig(drift_js_threshold=1.1).validate()


class GateTests(unittest.TestCase):
    def test_rest_does_not_fire(self):
        gate = IdleGate()
        d = gate.update(rest(), 0)
        self.assertFalse(d.fired)
        self.assertEqual(d.state, "idle")

    def test_single_strong_window_does_not_fire(self):
        gate = IdleGate()
        d = gate.update(strong("right"), 0)
        self.assertFalse(d.fired)
        self.assertEqual(d.state, "candidate")

    def test_three_sustained_windows_fire(self):
        gate = IdleGate()
        decisions = [gate.update(strong("right"), i * 100) for i in range(3)]
        self.assertTrue(decisions[-1].fired)
        self.assertEqual(decisions[-1].command, "right")

    def test_label_change_resets_dwell(self):
        gate = IdleGate()
        gate.update(strong("right"), 0)
        gate.update(strong("left"), 100)
        d = gate.update(strong("left"), 200)
        self.assertFalse(d.fired)
        self.assertEqual(d.state, "candidate")

    def test_low_probability_resets_candidate(self):
        gate = IdleGate()
        gate.update(strong("right"), 0)
        d = gate.update(pred(0.4, 0.4, 0.2), 100)
        self.assertEqual(d.state, "idle")
        self.assertFalse(d.fired)

    def test_low_margin_does_not_fire(self):
        gate = IdleGate(GateConfig(min_margin=0.4))
        ds = [gate.update(pred(0.3, 0.55, 0.15), i * 100) for i in range(4)]
        self.assertTrue(all(not d.fired for d in ds))

    def test_refractory_suppresses_repeat(self):
        gate = IdleGate(GateConfig(refractory_ms=1000))
        for i in range(3): d = gate.update(strong("right"), i * 100)
        self.assertTrue(d.fired)
        d = gate.update(strong("right"), 300)
        self.assertEqual(d.state, "refractory")
        self.assertFalse(d.fired)

    def test_neutral_rearms_after_fire(self):
        gate = IdleGate(GateConfig(refractory_ms=5000))
        for i in range(3): gate.update(strong("right"), i * 100)
        gate.update(rest(), 400)
        ds = [gate.update(strong("left"), 500 + i * 100) for i in range(3)]
        self.assertTrue(ds[-1].fired)
        self.assertEqual(ds[-1].command, "left")

    def test_drift_blocks_runaway_stream(self):
        gate = IdleGate(GateConfig(drift_window=8, drift_js_threshold=0.2))
        for i in range(8): gate.update(rest(), i * 100)
        ds = [gate.update(strong("right", 0.95), 800 + i * 100) for i in range(8)]
        self.assertTrue(any(d.state == "blocked_drift" for d in ds))

    def test_drift_can_be_disabled(self):
        cfg = GateConfig(drift_window=8, drift_js_threshold=0.01, block_on_drift=False)
        gate = IdleGate(cfg)
        for i in range(8): gate.update(rest(), i * 100)
        ds = [gate.update(strong("right", 0.95), 800 + i * 100) for i in range(3)]
        self.assertTrue(ds[-1].fired)

    def test_probability_sum_validated(self):
        with self.assertRaises(ValueError): IdleGate().update(pred(0.2, 0.2, 0.2), 0)

    def test_probability_range_validated(self):
        with self.assertRaises(ValueError): IdleGate().update({"probabilities": {"a": 1.2, "b": -0.2}}, 0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): IdleGate().update({"probabilities": {"a": float("nan"), "b": 1}}, 0)

    def test_requires_two_classes(self):
        with self.assertRaises(ValueError): IdleGate().update({"probabilities": {"a": 1.0}}, 0)

    def test_audit_hash_changes(self):
        gate = IdleGate()
        a = gate.update(rest(), 0).audit_hash
        b = gate.update(rest(), 100).audit_hash
        self.assertNotEqual(a, b)
        self.assertEqual(len(b), 64)

    def test_snapshot_contains_config_and_hash(self):
        gate = IdleGate()
        gate.update(rest(), 0)
        snap = gate.snapshot()
        self.assertIn("config", snap)
        self.assertIn("audit_hash", snap)

    def test_reset_clears_state(self):
        gate = IdleGate()
        gate.update(strong("right"), 0)
        gate.reset()
        snap = gate.snapshot()
        self.assertIsNone(snap["candidate"])
        self.assertEqual(snap["history"], [])


class MathTests(unittest.TestCase):
    def test_js_divergence_identical_zero(self):
        self.assertEqual(js_divergence({"a": 1}, {"a": 1}), 0)

    def test_js_divergence_disjoint_one(self):
        self.assertAlmostEqual(js_divergence({"a": 1}, {"b": 1}), 1.0)

    def test_js_divergence_symmetric(self):
        a = {"a": .7, "b": .3}; b = {"a": .4, "b": .6}
        self.assertAlmostEqual(js_divergence(a,b), js_divergence(b,a))


class AdapterTests(unittest.TestCase):
    def test_dydema_prediction_normalized(self):
        item = {"object":"classification","index":0,"label":"right","probabilities":{"left":.21,"right":.79}}
        self.assertEqual(from_dydema_prediction(item)["probabilities"]["right"], .79)

    def test_dydema_label_must_exist(self):
        with self.assertRaises(ValueError): from_dydema_prediction({"label":"up","probabilities":{"left":1}})

    def test_dydema_batch(self):
        rows = batch_from_dydema_response({"object":"list","data":[{"object":"classification","label":"right","probabilities":{"left":.2,"right":.8}}]})
        self.assertEqual(len(rows), 1)

    def test_dydema_batch_requires_list(self):
        with self.assertRaises(ValueError): batch_from_dydema_response({"data": {}})


class EvaluationTests(unittest.TestCase):
    def test_evaluate_counts_false_fires(self):
        events = [StreamEvent("rest", strong("right", .9)["probabilities"], i*100) for i in range(4)]
        report = evaluate_stream(events, GateConfig(dwell_windows=2, drift_window=4, block_on_drift=False))
        self.assertEqual(report["fires"], 1)
        self.assertEqual(report["false_fires"], 1)

    def test_evaluate_correct_fire(self):
        events = [StreamEvent("right", strong("right", .9)["probabilities"], i*100) for i in range(3)]
        report = evaluate_stream(events, GateConfig(block_on_drift=False))
        self.assertEqual(report["correct_fires"], 1)
        self.assertEqual(report["false_fires"], 0)

    def test_latency_reported(self):
        events = [StreamEvent("right", strong("right", .9)["probabilities"], i*100) for i in range(3)]
        report = evaluate_stream(events, GateConfig(block_on_drift=False))
        self.assertEqual(report["mean_fire_latency_ms"], 200)

    def test_synthetic_demo_no_false_fire(self):
        report = evaluate_stream(synthetic_demo())
        self.assertEqual(report["false_fires"], 0)
        self.assertGreaterEqual(report["fires"], 2)
        self.assertGreater(report["blocked_windows"], 0)

    def test_calibrate_returns_config(self):
        result = calibrate_threshold(synthetic_demo(100), target_false_per_minute=0)
        self.assertIn("enter_probability", result)
        self.assertIn("report", result)

    def test_calibrate_rejects_negative_target(self):
        with self.assertRaises(ValueError): calibrate_threshold([], -1)


class CLITests(unittest.TestCase):
    def test_simulate_cli(self):
        proc = subprocess.run([sys.executable, "-m", "idle_gate.cli", "simulate"], cwd=Path(__file__).parents[1], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["false_fires"], 0)

    def test_gate_cli_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"events.jsonl"
            p.write_text('\n'.join(json.dumps({"time_ms":i*100,"truth":"right","probabilities":strong("right",.9)["probabilities"]}) for i in range(3)))
            proc = subprocess.run([sys.executable, "-m", "idle_gate.cli", "gate", str(p)], cwd=Path(__file__).parents[1], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(json.loads(proc.stdout)["decisions"][-1]["fired"])


if __name__ == "__main__":
    unittest.main()
