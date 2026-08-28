import pytest

from convxai.engine import get_engine


@pytest.mark.parametrize("dataset_key", ["iris", "breast_cancer"])
@pytest.mark.parametrize("model_key", ["logistic_regression", "random_forest"])
def test_prediction_and_explanation_are_reproducible(dataset_key: str, model_key: str) -> None:
    engine = get_engine(dataset_key, model_key)
    first = engine.predict(0)
    second = engine.predict(0)
    method, effects = engine.local_explanation(0)

    assert first == second
    assert 0.0 <= first.probability <= 1.0
    assert method
    assert len(effects) == 3
    assert len({effect.feature for effect in effects}) == 3


def test_what_if_validates_feature_and_record() -> None:
    engine = get_engine("iris", "logistic_regression")
    result = engine.what_if(0, "sepal length", 6.2)
    assert result.feature == "sepal length (cm)"
    assert result.new_value == 6.2

    with pytest.raises(IndexError):
        engine.predict(10_000)
    with pytest.raises(KeyError):
        engine.what_if(0, "unknown feature", 1.0)


def test_global_importance_is_ranked() -> None:
    engine = get_engine("breast_cancer", "random_forest")
    importance = engine.global_importance()
    assert len(importance) == 5
    assert [item.effect for item in importance] == sorted(
        [item.effect for item in importance], reverse=True
    )
