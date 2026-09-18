"""Configuration loading and project paths.

All tunable values live in ``config/*.yaml``; modules read them from here so a
number is defined in exactly one place.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pipeline_config() -> dict:
    return load_yaml("pipeline.yaml")


def cost_config() -> dict:
    return load_yaml("costs.yaml")


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    models: Path
    outputs: Path

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def tables(self) -> Path:
        return self.outputs / "tables"

    @property
    def results(self) -> Path:
        return self.outputs / "results.json"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.models,
                  self.figures, self.tables):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths(cfg: dict | None = None) -> Paths:
    cfg = cfg or pipeline_config()
    p = cfg["paths"]
    return Paths(
        raw=ROOT / p["raw_dir"],
        interim=ROOT / p["interim_dir"],
        processed=ROOT / p["processed_dir"],
        models=ROOT / p["model_dir"],
        outputs=ROOT / p["outputs_dir"],
    )


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches. Modules still pass explicit seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
