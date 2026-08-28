# ConvXAI

[![CI](https://github.com/Naviden/ConvXAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Naviden/ConvXAI/actions/workflows/ci.yml)

ConvXAI is a reproducible reference implementation of **conversational explainable AI**:
instead of navigating an explainer dashboard, a user selects a dataset and model and asks
questions about predictions in plain language.

This repository contains two generations of the project:

- `src/convxai/` is the maintained reference implementation documented below.
- `RASA_NLU/` and `chat-app/` contain the original 2021 research prototype for provenance.
  They depend on an end-of-life Rasa/TensorFlow stack and are not the supported run path.

## What the reference implementation supports

- Built-in Iris and Breast Cancer datasets from scikit-learn; no data download is required.
- Deterministic logistic-regression and random-forest models.
- Independent conversation sessions instead of process-wide pickle state.
- Prediction, local explanation, global feature importance, and numerical what-if questions.
- Audience-aware responses for researchers, domain experts, and general users.
- A same-origin web interface served by the Python application; no Node toolchain is needed.
- Automated tests and a timed reproduction command that records versions and metrics.

The local explanations are deliberately transparent:

- Logistic regression uses exact standardized logit contributions for the predicted class.
- Random forest uses deterministic feature ablation to the training mean and measures the
  resulting change in predicted-class probability.

These are reference explanations, not causal claims. See [Scientific scope](#scientific-scope).

## Quick start with `uv`

Python 3.11–3.13 is supported. Python 3.12 is used for the reference lock and CI path.

```bash
uv sync --locked --extra dev
uv run convxai serve
```

Open <http://127.0.0.1:8000> and try:

```text
why record 3
predict record 8
list features
global importance
what if record 3 sepal length is 6.2
```

The first three messages select the dataset, model, and audience profile. Type `reset` at any
time to start a new conversation.

## Standard virtual-environment setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
convxai serve
```

## Reproduce the reference experiment

```bash
uv run python scripts/reproduce.py --output artifacts/reproduction.json
```

The command trains every supported dataset/model pair, records accuracy and fit time, checks a
representative local explanation, and writes the complete environment and wall-clock duration to
JSON. Random seeds and model parallelism are fixed for deterministic results.

With the locked Python 3.12 environment, seed 42, and the fixed stratified 75/25 splits, the
expected held-out accuracies are:

| Dataset | Model | Accuracy |
| --- | --- | ---: |
| Iris | Logistic regression | 0.9211 |
| Iris | Random forest | 0.8947 |
| Breast Cancer Wisconsin | Logistic regression | 0.9860 |
| Breast Cancer Wisconsin | Random forest | 0.9580 |

The test suite checks these values to four decimal places. Execution times are recorded for
traceability but are hardware-dependent and are not treated as correctness criteria.

## Tests and quality checks

```bash
uv run pytest
uv run ruff check src tests scripts
```

## Architecture

```text
Browser UI
   │  JSON over same-origin HTTP
   ▼
FastAPI routes ──► per-session dialogue state ──► deterministic intent parser
                                                     │
                                                     ▼
Dataset catalog ──► trained model cache ──► prediction/explanation engine
```

The implementation intentionally avoids serialized models and executable pickle files. Models are
small and are retrained deterministically from built-in datasets when first requested.

## Scientific scope

ConvXAI demonstrates how explanations can be requested and presented conversationally. It is not a
claim that conversational delivery makes an explanation faithful, causal, fair, or useful. In
particular:

- Feature ablation can create unrealistic samples and should be interpreted as a sensitivity probe.
- Logistic-regression contributions are model-specific associations, not causal effects.
- Audience profiles currently change wording and level of detail, not the explanation algorithm.
- The bundled datasets are pedagogical and are not evidence of clinical or operational validity.

Research extensions should evaluate explanation fidelity, calibration, user understanding,
appropriate reliance, and robustness to paraphrases and adversarial inputs.

## Legacy prototype

The historical prototype combined Rasa 2.2, TensorFlow 2.3, Flask, React 17, LIME, SHAP, and several
pretrained NLU models. Its frozen Python 3.7 environment and generated model artifacts are retained
only to make the project history inspectable. Do not deploy the legacy servers or load their pickle
files in an untrusted environment.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). ConvXAI is released under the MIT
License; see [`LICENSE`](LICENSE).
