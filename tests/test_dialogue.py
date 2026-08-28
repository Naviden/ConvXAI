from convxai.dialogue import SessionStore


def configured_session() -> tuple[SessionStore, str]:
    store = SessionStore()
    session, _ = store.create()
    store.respond(session.identifier, "Iris")
    store.respond(session.identifier, "Random forest")
    store.respond(session.identifier, "Researcher")
    return store, session.identifier


def test_configuration_and_local_explanation() -> None:
    store, identifier = configured_session()
    response = store.respond(identifier, "why record 3")
    assert "predicted as" in response.message
    assert response.data["method"] == "local mean-ablation sensitivity"
    assert len(response.data["features"]) == 3


def test_what_if_and_reset() -> None:
    store, identifier = configured_session()
    response = store.respond(identifier, "what if record 3 sepal length is 6.2")
    assert "sensitivity result" in response.message

    reset = store.respond(identifier, "reset")
    assert "Choose a dataset" in reset.message


def test_unknown_session_is_rejected() -> None:
    store = SessionStore()
    try:
        store.respond("missing", "hello")
    except KeyError as error:
        assert "Unknown session" in str(error)
    else:
        raise AssertionError("Unknown session should fail")


def test_sessions_keep_independent_configuration() -> None:
    store = SessionStore()
    first, _ = store.create()
    second, _ = store.create()

    store.respond(first.identifier, "Iris")
    store.respond(second.identifier, "Breast Cancer Wisconsin")

    assert store.get(first.identifier).dataset_key == "iris"
    assert store.get(second.identifier).dataset_key == "breast_cancer"
