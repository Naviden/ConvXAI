from fastapi.testclient import TestClient

from convxai.api import app

client = TestClient(app)


def test_health_and_ui() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    page = client.get("/")
    assert page.status_code == 200
    assert "ConvXAI" in page.text


def test_conversation_api() -> None:
    created = client.post("/api/session")
    assert created.status_code == 200
    identifier = created.json()["session_id"]

    for message in ("Iris", "Logistic regression", "Researcher"):
        response = client.post("/api/message", json={"session_id": identifier, "text": message})
        assert response.status_code == 200

    explanation = client.post(
        "/api/message", json={"session_id": identifier, "text": "why record 2"}
    )
    assert explanation.status_code == 200
    assert explanation.json()["response"]["data"]["method"]

    global_explanation = client.post(
        "/api/message",
        json={"session_id": identifier, "text": "global importance"},
    )
    assert global_explanation.status_code == 200
    features = global_explanation.json()["response"]["data"]["features"]
    assert features
    assert features[0]["value"] is None


def test_unknown_session_returns_404() -> None:
    response = client.post("/api/message", json={"session_id": "missing", "text": "hello"})
    assert response.status_code == 404
