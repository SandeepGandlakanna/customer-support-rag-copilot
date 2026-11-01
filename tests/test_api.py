
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.config import APP_API_KEY


client = TestClient(app)

AUTH_HEADERS = {
    "X-API-Key": APP_API_KEY
}


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint_success():

    fake_result = {
        "answer": "You can request a refund within 30 days.",
        "sources": [
            {
                "source": "support_faq.txt",
                "section": "1. What is the refund policy?"
            }
        ],
        "status": "answered"
    }

    with patch(
        "app.main.ask_question",
        return_value=fake_result
    ):

        response = client.post(
            "/ask",
            json={
                "question": "What is the refund policy?"
            },
            headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert response.json() == fake_result


def test_ask_endpoint_rejects_empty_question():

    response = client.post(
        "/ask",
        json={
            "question": ""
        },
        headers=AUTH_HEADERS
    )

    assert response.status_code == 422


def test_ask_endpoint_rejects_long_question():

    response = client.post(
        "/ask",
        json={
            "question": "a" * 2001
        },
        headers=AUTH_HEADERS
    )

    assert response.status_code == 422


def test_ask_endpoint_handles_internal_error():

    with patch(
        "app.main.ask_question",
        side_effect=Exception("Unexpected error")
    ):

        response = client.post(
            "/ask",
            json={
                "question": "What is the refund policy?"
            },
            headers=AUTH_HEADERS
        )

    assert response.status_code == 500


def test_ask_endpoint_returns_fallback_status():

    fake_result = {
        "answer": "I don't have enough information.",
        "sources": [],
        "status": "fallback"
    }

    with patch(
        "app.main.ask_question",
        return_value=fake_result
    ):

        response = client.post(
            "/ask",
            json={
                "question": "Who won the football world cup?"
            },
            headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert response.json()["status"] == "fallback"


def test_ask_endpoint_returns_error_status():

    fake_result = {
        "answer": "Unable to process your request.",
        "sources": [],
        "status": "error"
    }

    with patch(
        "app.main.ask_question",
        return_value=fake_result
    ):

        response = client.post(
            "/ask",
            json={
                "question": "What is the refund policy?"
            },
            headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_ask_endpoint_rejects_missing_api_key():

    response = client.post(
        "/ask",
        json={
            "question": "What is the refund policy?"
        }
    )

    assert response.status_code == 422


def test_ask_endpoint_rejects_invalid_api_key():

    response = client.post(
        "/ask",
        json={
            "question": "What is the refund policy?"
        },
        headers={
            "X-API-Key": "invalid-test-key"
        }
    )

    assert response.status_code == 401


def test_ask_endpoint_rejects_whitespace_question():

    response = client.post(
        "/ask",
        json={
            "question": "     "
        },
        headers=AUTH_HEADERS
    )

    assert response.status_code == 422
