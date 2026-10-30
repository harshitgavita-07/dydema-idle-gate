# Dydema Idle Gate

A small command-safety prototype for Dydema's documented EEG classifier boundary: **every prediction returns a class, including during rest, and there is no automatic idle detector**.

That sentence is from Dydema's own zero-shot/device-command docs. If an app turns every returned class into a key press or agent command, rest windows and noisy probability ties can become unwanted actions. This repo adds the missing application-side abstention layer before an action runner.

It is a prototype, not a claim that Wizard is inaccurate, not a medical or clinical tool, and not an official Dydema integration.

## What it does

- Accepts Dydema-style `{label, probabilities}` prediction objects.
- Requires a configurable probability threshold and winner margin.
- Requires the same command to stay above threshold for several windows before firing.
- Treats `rest`, `idle` and `neutral` as abstention classes and uses them to re-arm after a command.
- Suppresses repeated commands during a refractory period.
- Detects label-distribution drift with Jensen-Shannon divergence and blocks commands when a stream starts running away.
- Calibrates threshold, margin and dwell against a target false-activation rate.
- Reports false activations per minute, command fire rate, fire latency and blocked windows.
- Writes a SHA-256 hash-chained audit record for every decision.
- Has no runtime dependencies and makes no network calls.

## Why this is tied to Dydema

Dydema's Wizard API returns a label and class probabilities per trial. The docs are careful to say that the hosted task model does not decide when *not* to act. That split makes sense for a model API, but every game, keyboard, device or agent runner still needs a local policy between probabilities and effects.

This prototype is that policy layer. It does not replace model training or Dydema's API; it sits after `POST /v1/task-models/{id}/predictions` and before a keypress, device command or agent tool call.

## Demo

```sh
python -m unittest discover -s tests -v
python -m idle_gate.cli simulate
```

The deterministic demo contains rest, weak command evidence, strong left/right commands and then a runaway right-command stream. Current result:

- 98 prediction windows
- 3 correct fired commands
- 0 false activations
- 27 drift-blocked windows
- 0 false activations/minute
- mean fire latency 666.7 ms at 250 ms/window

See [`examples/simulation-report.json`](examples/simulation-report.json).

## Use with a Dydema response

```python
from idle_gate import IdleGate
from idle_gate.adapter import from_dydema_prediction

# One item from Dydema's prediction response["data"].
prediction = from_dydema_prediction({
    "object": "classification",
    "index": 0,
    "label": "right",
    "probabilities": {"left": 0.10, "right": 0.84, "rest": 0.06},
})

decision = gate.update(prediction)
if decision.fired:
    # Only here should an app press a key or invoke an agent command.
    send_command(decision.command)
```

The adapter validates the documented response shape. It never sends EEG, labels, prompts or API keys anywhere.

## JSONL evaluation

Each line can carry a truth label for offline evaluation:

```json
{"time_ms": 0, "truth": "right", "probabilities": {"left": 0.08, "right": 0.86, "rest": 0.06}}
```

Run:

```sh
python -m idle_gate.cli evaluate events.jsonl
python -m idle_gate.cli calibrate events.jsonl --target-false-per-minute 0.2
python -m idle_gate.cli gate events.jsonl
```

Calibration searches probability, margin and dwell settings. It reports the best feasible setting, preferring command recall, then fewer false activations, then lower latency.

## Design choice

A returned class is evidence, not intent. The gate changes the contract from:

```text
classifier label -> action
```

to:

```text
class probabilities -> sustained evidence -> one action -> refractory/re-arm
```

and adds a drift stop for streams whose label distribution moves sharply away from their baseline.

## Boundaries

- Tested with deterministic synthetic Dydema-response-shaped windows only. I did not use a headset, private EEG data, a Dydema API key, or paid API calls.
- No accuracy claim about Wizard or any headset. A real deployment should calibrate per user/session and evaluate on held-out recordings.
- Drift blocking is conservative and can block valid commands after a real distribution shift; an operator should review or recalibrate after a block.
- The audit hash is tamper evidence for a local log, not a security boundary against a malicious writer.
- This does not provide stimulus design, ERP aggregation, medical safety, or consent handling. Dydema's research-use and acceptable-use limits still apply.

## Sources

- Dydema home: <https://dydema.com>
- Dydema about/team: <https://dydema.com/about>
- Dydema developer docs: <https://developers.dydema.com>
- Wizard/task model docs: <https://developers.dydema.com/docs/task-models>
- Quickstart: <https://developers.dydema.com/docs/quickstart>
- Contact page: <https://dydema.com/contact>
- Julian Weaver's site: <https://juliver.xyz>

AI-assisted implementation and tests. Built independently from public docs. Not affiliated with Dydema.
