import json

from convxai.reproduce import run


def test_reproduction_writes_timed_results(tmp_path) -> None:
    output = tmp_path / "reproduction.json"
    payload = run(output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert len(payload["experiments"]) == 4
    assert payload["total_seconds"] >= 0
    assert all(item["experiment_seconds"] >= 0 for item in payload["experiments"])
    assert persisted["random_seed"] == 42
    accuracies = {
        (item["dataset"], item["model"]): round(item["accuracy"], 4)
        for item in payload["experiments"]
    }
    assert accuracies == {
        ("iris", "logistic_regression"): 0.9211,
        ("iris", "random_forest"): 0.8947,
        ("breast_cancer", "logistic_regression"): 0.986,
        ("breast_cancer", "random_forest"): 0.958,
    }
