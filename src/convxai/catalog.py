"""Deterministic, download-free datasets used by ConvXAI."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


@dataclass(frozen=True)
class DatasetBundle:
    key: str
    display_name: str
    description: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


DATASET_LABELS = {
    "iris": "Iris",
    "breast_cancer": "Breast Cancer Wisconsin",
}


def normalize_choice(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def resolve_dataset(value: str) -> str | None:
    normalized = normalize_choice(value)
    aliases = {
        "iris": "iris",
        "breast_cancer": "breast_cancer",
        "breast_cancer_wisconsin": "breast_cancer",
        "cancer": "breast_cancer",
    }
    return aliases.get(normalized)


@cache
def load_dataset(key: str) -> DatasetBundle:
    if key == "iris":
        raw = load_iris()
        description = "Three-class flower classification from four morphological measurements."
    elif key == "breast_cancer":
        raw = load_breast_cancer()
        description = "Binary diagnostic classification from computed cell-nucleus features."
    else:
        raise KeyError(f"Unsupported dataset: {key}")

    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(raw.data, dtype=float),
        np.asarray(raw.target, dtype=int),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=raw.target,
    )
    return DatasetBundle(
        key=key,
        display_name=DATASET_LABELS[key],
        description=description,
        feature_names=tuple(str(name) for name in raw.feature_names),
        target_names=tuple(str(name) for name in raw.target_names),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
