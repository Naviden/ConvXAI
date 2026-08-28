"""Deterministic prediction and explanation engine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from time import perf_counter

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from convxai.catalog import RANDOM_SEED, DatasetBundle, load_dataset, normalize_choice


@dataclass(frozen=True)
class Prediction:
    record: int
    class_index: int
    class_name: str
    probability: float
    true_class_name: str


@dataclass(frozen=True)
class FeatureEffect:
    feature: str
    value: float | None
    effect: float
    direction: str


@dataclass(frozen=True)
class WhatIfResult:
    feature: str
    old_value: float
    new_value: float
    original: Prediction
    changed_class_name: str
    changed_probability: float


MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}


def resolve_model(value: str) -> str | None:
    normalized = normalize_choice(value)
    aliases = {
        "logistic": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "random_forest": "random_forest",
        "forest": "random_forest",
        "rf": "random_forest",
    }
    return aliases.get(normalized)


class ExplanationEngine:
    def __init__(self, dataset: DatasetBundle, model_key: str) -> None:
        self.dataset = dataset
        self.model_key = model_key
        self.fit_seconds = 0.0
        self.model = self._fit_model()

    def _fit_model(self) -> Pipeline | ClassifierMixin:
        started = perf_counter()
        if self.model_key == "logistic_regression":
            model: Pipeline | ClassifierMixin = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED),
                    ),
                ]
            )
        elif self.model_key == "random_forest":
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
        else:
            raise KeyError(f"Unsupported model: {self.model_key}")

        model.fit(self.dataset.X_train, self.dataset.y_train)
        self.fit_seconds = perf_counter() - started
        return model

    @property
    def accuracy(self) -> float:
        predictions = self.model.predict(self.dataset.X_test)
        return float(accuracy_score(self.dataset.y_test, predictions))

    def _record(self, record: int) -> np.ndarray:
        if record < 0 or record >= len(self.dataset.X_test):
            raise IndexError(f"Record must be between 0 and {len(self.dataset.X_test) - 1}.")
        return self.dataset.X_test[record]

    def predict(self, record: int) -> Prediction:
        row = self._record(record)
        class_index = int(self.model.predict(row.reshape(1, -1))[0])
        probability = float(self.model.predict_proba(row.reshape(1, -1))[0, class_index])
        true_index = int(self.dataset.y_test[record])
        return Prediction(
            record=record,
            class_index=class_index,
            class_name=self.dataset.target_names[class_index],
            probability=probability,
            true_class_name=self.dataset.target_names[true_index],
        )

    def local_explanation(self, record: int, top_k: int = 3) -> tuple[str, list[FeatureEffect]]:
        row = self._record(record)
        prediction = self.predict(record)

        if self.model_key == "logistic_regression":
            pipeline = self.model
            assert isinstance(pipeline, Pipeline)
            scaler = pipeline.named_steps["scaler"]
            classifier = pipeline.named_steps["classifier"]
            standardized = scaler.transform(row.reshape(1, -1))[0]
            if len(classifier.classes_) == 2:
                sign = 1.0 if prediction.class_index == int(classifier.classes_[1]) else -1.0
                effects = sign * classifier.coef_[0] * standardized
            else:
                class_position = list(classifier.classes_).index(prediction.class_index)
                effects = classifier.coef_[class_position] * standardized
            method = "standardized logit contribution"
        else:
            baseline_probability = prediction.probability
            effects = np.zeros(len(row), dtype=float)
            training_means = self.dataset.X_train.mean(axis=0)
            for index in range(len(row)):
                ablated = row.copy()
                ablated[index] = training_means[index]
                changed_probability = self.model.predict_proba(ablated.reshape(1, -1))[
                    0, prediction.class_index
                ]
                effects[index] = baseline_probability - changed_probability
            method = "local mean-ablation sensitivity"

        ranked = np.argsort(np.abs(effects))[::-1][:top_k]
        results = [
            FeatureEffect(
                feature=self.dataset.feature_names[index],
                value=float(row[index]),
                effect=float(effects[index]),
                direction="supports" if effects[index] >= 0 else "opposes",
            )
            for index in ranked
        ]
        return method, results

    def global_importance(self, top_k: int = 5) -> list[FeatureEffect]:
        if self.model_key == "logistic_regression":
            pipeline = self.model
            assert isinstance(pipeline, Pipeline)
            classifier = pipeline.named_steps["classifier"]
            importance = np.mean(np.abs(classifier.coef_), axis=0)
        else:
            importance = np.asarray(self.model.feature_importances_)

        ranked = np.argsort(importance)[::-1][:top_k]
        return [
            FeatureEffect(
                feature=self.dataset.feature_names[index],
                value=None,
                effect=float(importance[index]),
                direction="importance",
            )
            for index in ranked
        ]

    def resolve_feature(self, value: str) -> str:
        wanted = normalize_choice(value)
        normalized = {
            normalize_choice(feature.replace("(cm)", "")): feature
            for feature in self.dataset.feature_names
        }
        if wanted in normalized:
            return normalized[wanted]
        matches = [feature for key, feature in normalized.items() if wanted in key or key in wanted]
        if len(matches) == 1:
            return matches[0]
        choices = ", ".join(self.dataset.feature_names)
        raise KeyError(f"Unknown or ambiguous feature '{value}'. Choose one of: {choices}.")

    def what_if(self, record: int, feature: str, new_value: float) -> WhatIfResult:
        row = self._record(record)
        resolved = self.resolve_feature(feature)
        feature_index = self.dataset.feature_names.index(resolved)
        modified = row.copy()
        old_value = float(modified[feature_index])
        modified[feature_index] = new_value
        original = self.predict(record)
        changed_index = int(self.model.predict(modified.reshape(1, -1))[0])
        changed_probability = float(
            self.model.predict_proba(modified.reshape(1, -1))[0, changed_index]
        )
        return WhatIfResult(
            feature=resolved,
            old_value=old_value,
            new_value=new_value,
            original=original,
            changed_class_name=self.dataset.target_names[changed_index],
            changed_probability=changed_probability,
        )


@cache
def get_engine(dataset_key: str, model_key: str) -> ExplanationEngine:
    return ExplanationEngine(load_dataset(dataset_key), model_key)
