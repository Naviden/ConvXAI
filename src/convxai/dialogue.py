"""Session-safe conversational layer for the ConvXAI engine."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from convxai.catalog import DATASET_LABELS, load_dataset, resolve_dataset
from convxai.engine import MODEL_LABELS, FeatureEffect, get_engine, resolve_model

PROFILES = {
    "researcher": "Researcher",
    "domain_expert": "Domain expert",
    "general": "General audience",
}


def _normalize(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def resolve_profile(value: str) -> str | None:
    aliases = {
        "researcher": "researcher",
        "scientist": "researcher",
        "domain_expert": "domain_expert",
        "expert": "domain_expert",
        "general": "general",
        "general_audience": "general",
        "end_user": "general",
    }
    return aliases.get(_normalize(value))


@dataclass
class Session:
    identifier: str
    stage: str = "dataset"
    dataset_key: str | None = None
    model_key: str | None = None
    profile: str | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class DialogueResponse:
    message: str
    options: tuple[str, ...] = ()
    kind: str = "text"
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "options": list(self.options),
            "kind": self.kind,
            "data": self.data,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create(self) -> tuple[Session, DialogueResponse]:
        with self._lock:
            identifier = str(uuid.uuid4())
            session = Session(identifier=identifier)
            self._sessions[identifier] = session
        return session, self._dataset_prompt()

    def get(self, identifier: str) -> Session:
        with self._lock:
            try:
                return self._sessions[identifier]
            except KeyError as error:
                raise KeyError("Unknown session. Start a new conversation.") from error

    def reset(self, session: Session) -> DialogueResponse:
        with self._lock:
            session.stage = "dataset"
            session.dataset_key = None
            session.model_key = None
            session.profile = None
            session.transcript.clear()
        return self._dataset_prompt()

    @staticmethod
    def _dataset_prompt() -> DialogueResponse:
        return DialogueResponse(
            "Choose a dataset. No external data will be downloaded.",
            tuple(DATASET_LABELS.values()),
        )

    def respond(self, identifier: str, text: str) -> DialogueResponse:
        session = self.get(identifier)
        clean_text = text.strip()
        if not clean_text:
            return DialogueResponse("Please enter a question or select an option.")
        if _normalize(clean_text) in {"reset", "restart", "start_over"}:
            return self.reset(session)

        session.transcript.append({"role": "user", "text": clean_text})
        if session.stage == "dataset":
            response = self._choose_dataset(session, clean_text)
        elif session.stage == "model":
            response = self._choose_model(session, clean_text)
        elif session.stage == "profile":
            response = self._choose_profile(session, clean_text)
        else:
            response = self._answer_question(session, clean_text)
        session.transcript.append({"role": "assistant", "text": response.message})
        return response

    def _choose_dataset(self, session: Session, text: str) -> DialogueResponse:
        dataset_key = resolve_dataset(text)
        if dataset_key is None:
            return DialogueResponse(
                "I could not match that dataset.", tuple(DATASET_LABELS.values())
            )
        session.dataset_key = dataset_key
        session.stage = "model"
        dataset = load_dataset(dataset_key)
        return DialogueResponse(
            f"Selected {dataset.display_name}: {dataset.description} Choose a model.",
            tuple(MODEL_LABELS.values()),
        )

    def _choose_model(self, session: Session, text: str) -> DialogueResponse:
        model_key = resolve_model(text)
        if model_key is None:
            return DialogueResponse("I could not match that model.", tuple(MODEL_LABELS.values()))
        session.model_key = model_key
        session.stage = "profile"
        return DialogueResponse(
            f"Selected {MODEL_LABELS[model_key]}. Who should the explanations be written for?",
            tuple(PROFILES.values()),
        )

    def _choose_profile(self, session: Session, text: str) -> DialogueResponse:
        profile = resolve_profile(text)
        if profile is None:
            return DialogueResponse("I could not match that audience.", tuple(PROFILES.values()))
        session.profile = profile
        session.stage = "ready"
        assert session.dataset_key and session.model_key
        engine = get_engine(session.dataset_key, session.model_key)
        return DialogueResponse(
            f"Ready. The held-out accuracy is {engine.accuracy:.3f}. "
            "Ask 'help' to see supported questions.",
            ("why record 3", "predict record 3", "global importance", "list features"),
            data={"accuracy": engine.accuracy, "fit_seconds": engine.fit_seconds},
        )

    def _answer_question(self, session: Session, text: str) -> DialogueResponse:
        assert session.dataset_key and session.model_key and session.profile
        engine = get_engine(session.dataset_key, session.model_key)
        normalized = text.lower().strip()

        if normalized in {"help", "?"} or "what can" in normalized:
            return DialogueResponse(
                "Supported questions: predict record N; why record N; "
                "most important feature for record N; "
                "global importance; list features; what if record N FEATURE is VALUE; reset.",
                ("why record 3", "global importance", "what if record 3 sepal length is 6.2"),
            )

        if "list features" in normalized or normalized == "features":
            features = ", ".join(engine.dataset.feature_names)
            return DialogueResponse(f"Features: {features}.")

        if "global" in normalized and ("importance" in normalized or "important" in normalized):
            effects = engine.global_importance()
            lines = [f"{item.feature}: {item.effect:.4f}" for item in effects]
            return DialogueResponse(
                "Global feature importance, highest first:\n" + "\n".join(lines),
                data={"features": [self._effect_dict(item) for item in effects]},
            )

        what_if = re.search(
            r"what\s+if\s+(?:record|instance)\s+(\d+)\s+(.+?)\s+(?:is|=|to)\s+(-?\d+(?:\.\d+)?)\s*$",
            normalized,
        )
        if what_if:
            record = int(what_if.group(1))
            feature = what_if.group(2).strip()
            value = float(what_if.group(3))
            try:
                result = engine.what_if(record, feature, value)
            except (IndexError, KeyError) as error:
                return DialogueResponse(str(error))
            changed = result.original.class_name != result.changed_class_name
            return DialogueResponse(
                f"Record {record}: changing {result.feature} from {result.old_value:.3f} to "
                f"{result.new_value:.3f} {'changes' if changed else 'keeps'} the prediction from "
                f"{result.original.class_name} ({result.original.probability:.3f}) to "
                f"{result.changed_class_name} ({result.changed_probability:.3f}). This is a model "
                "sensitivity result, not a causal effect.",
                data={
                    "changed": changed,
                    "original_class": result.original.class_name,
                    "changed_class": result.changed_class_name,
                },
            )

        record_match = re.search(r"(?:record|instance)\s+(\d+)", normalized)
        if record_match and any(word in normalized for word in ("why", "explain", "important")):
            record = int(record_match.group(1))
            try:
                prediction = engine.predict(record)
                method, effects = engine.local_explanation(record)
            except IndexError as error:
                return DialogueResponse(str(error))
            explanation = self._format_local(session.profile, method, effects)
            return DialogueResponse(
                f"Record {record} is predicted as {prediction.class_name} with probability "
                f"{prediction.probability:.3f} (held-out label: {prediction.true_class_name}).\n"
                f"{explanation}",
                data={
                    "prediction": prediction.class_name,
                    "probability": prediction.probability,
                    "true_class": prediction.true_class_name,
                    "method": method,
                    "features": [self._effect_dict(item) for item in effects],
                },
            )

        if record_match and any(word in normalized for word in ("predict", "class", "label")):
            record = int(record_match.group(1))
            try:
                prediction = engine.predict(record)
            except IndexError as error:
                return DialogueResponse(str(error))
            return DialogueResponse(
                f"Record {record} is predicted as {prediction.class_name} with probability "
                f"{prediction.probability:.3f}. Its held-out label is "
                f"{prediction.true_class_name}.",
                data={
                    "prediction": prediction.class_name,
                    "probability": prediction.probability,
                    "true_class": prediction.true_class_name,
                },
            )

        return DialogueResponse(
            "I could not map that question to a supported explanation. Ask 'help' for examples.",
            ("why record 3", "predict record 3", "global importance", "list features"),
        )

    @staticmethod
    def _effect_dict(item: FeatureEffect) -> dict[str, Any]:
        return {
            "feature": item.feature,
            "value": item.value,
            "effect": item.effect,
            "direction": item.direction,
        }

    @staticmethod
    def _format_local(profile: str, method: str, effects: list[FeatureEffect]) -> str:
        if profile == "general":
            names = ", ".join(item.feature for item in effects)
            return f"The most influential measured features were {names}."
        lines = [
            f"- {item.feature}={item.value:.3f}: {item.direction} the prediction "
            f"(effect={item.effect:+.4f})"
            for item in effects
        ]
        prefix = "Local explanation" if profile == "domain_expert" else f"Method: {method}"
        return prefix + "\n" + "\n".join(lines)


session_store = SessionStore()
