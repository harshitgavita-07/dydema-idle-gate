# Public documentation evidence - October 31, 2026

Dydema's developer documentation says Wizard returns a label and class probabilities for each trial.

In the zero-shot/device-command section, the docs state:

> Every trial returns a class, including during rest: there is no automatic idle detector.

Source checked: https://developers.dydema.com/docs/task-models

This prototype addresses that documented application boundary. It does not treat the behavior as a bug and does not test Dydema's paid API.
