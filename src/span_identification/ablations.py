"""Ablation config helpers for span identification experiments."""
from __future__ import annotations

from typing import Any


def build_ablation_configs(base_config: dict) -> list[dict]:
    """
    Build list of configs for ablation studies from base config.
    Varies label_scheme, etc. if specified in config.
    """
    configs = [base_config.copy()]
    ablations = base_config.get("ablations", {})
    if not ablations:
        return configs

    for key, values in ablations.items():
        new_configs = []
        for cfg in configs:
            for v in values:
                c = cfg.copy()
                if key == "label_scheme":
                    c["label_schemes"] = [v]
                elif key == "context_window":
                    c.setdefault("model", {})["max_length"] = v
                else:
                    c[key] = v
                new_configs.append(c)
        configs = new_configs
    return configs
