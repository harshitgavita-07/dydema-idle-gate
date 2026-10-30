from __future__ import annotations

from typing import Any, Mapping


def from_dydema_prediction(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one item from POST /v1/task-models/{id}/predictions.

    Expected shape: {"object":"classification", "index":0, "label":"right",
    "probabilities":{"left":0.21,"right":0.79}}
    """
    if item.get("object") not in {None, "classification"}:
        raise ValueError("not a Dydema classification object")
    probabilities = item.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError("classification has no probabilities")
    label = item.get("label")
    if label is not None and label not in probabilities:
        raise ValueError("returned label is missing from probabilities")
    return {"probabilities": dict(probabilities)}


def batch_from_dydema_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("response must contain a data list")
    return [from_dydema_prediction(item) for item in data]
