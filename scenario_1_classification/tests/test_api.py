from types import SimpleNamespace
from fastapi.testclient import TestClient
from app import create_app
from src.core.errors import ModelNotReadyError, StorageError

def fail(error):
    def operation(*args):
        raise error
    return operation

def container(error=None):
    return SimpleNamespace(artifacts=SimpleNamespace(ready=lambda: False),
        prediction=SimpleNamespace(predict=fail(error or ModelNotReadyError("Train first."))))

def test_model_not_ready_has_stable_error_contract():
    with TestClient(create_app(container=container())) as client:
        response = client.post("/predict", json={"message": "invoice issue"})
    assert response.status_code == 409
    assert response.json()["code"] == "model_not_ready"
    assert response.headers["X-Request-ID"] == response.json()["request_id"]

def test_invalid_request_does_not_echo_input():
    with TestClient(create_app(container=container())) as client:
        response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_error"

def test_storage_failure_is_service_unavailable():
    with TestClient(create_app(container=container(StorageError("Model unreadable.")))) as client:
        response = client.post("/predict", json={"message": "invoice issue"})
    assert response.status_code == 503

def test_unexpected_error_does_not_leak_exception():
    with TestClient(create_app(container=container(RuntimeError("sensitive internals"))), raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"message": "invoice issue"})
    assert response.status_code == 500
    assert "sensitive internals" not in response.text
