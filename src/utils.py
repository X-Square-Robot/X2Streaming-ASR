"""Configuration helpers used by inference entry points."""

from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
