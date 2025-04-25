from __future__ import annotations


def _values_processing(values: dict | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Process the parameter values."""
    values = values or dict()
    values_observables = values
    values_circuit = values
    val_keys = values.keys()
    if "circuit" in val_keys and "observables" in val_keys:
        values_observables = values["observables"]
        values_circuit = values["circuit"]
    return values_circuit, values_observables
