"""Timed deterministic reference experiment."""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from convxai import __version__
from convxai.catalog import DATASET_LABELS, RANDOM_SEED
from convxai.engine import MODEL_LABELS, get_engine

LOGGER = logging.getLogger("convxai.reproduce")


def run(output: Path) -> dict[str, Any]:
    started = perf_counter()
    results: list[dict[str, Any]] = []
    for dataset_key in DATASET_LABELS:
        for model_key in MODEL_LABELS:
            experiment_started = perf_counter()
            engine = get_engine(dataset_key, model_key)
            prediction = engine.predict(0)
            method, effects = engine.local_explanation(0)
            elapsed = perf_counter() - experiment_started
            LOGGER.info(
                "dataset=%s model=%s accuracy=%.4f elapsed_seconds=%.6f",
                dataset_key,
                model_key,
                engine.accuracy,
                elapsed,
            )
            results.append(
                {
                    "dataset": dataset_key,
                    "model": model_key,
                    "accuracy": engine.accuracy,
                    "fit_seconds": engine.fit_seconds,
                    "experiment_seconds": elapsed,
                    "reference_prediction": prediction.class_name,
                    "explanation_method": method,
                    "top_features": [effect.feature for effect in effects],
                }
            )

    payload = {
        "convxai_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_seed": RANDOM_SEED,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
        },
        "experiments": results,
        "total_seconds": perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote=%s total_seconds=%.6f", output, payload["total_seconds"])
    return payload
