"""Request ids are returned and logs stay free of secrets."""

import json
import logging

from app.logging import JsonFormatter, choose_request_id


def test_incoming_request_id_is_returned(client):
    response = client.get("/health/live", headers={"X-Request-ID": "review-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "review-123"


def test_missing_request_id_is_created(client):
    response = client.get("/health/live")
    request_id = response.headers["x-request-id"]
    assert request_id
    assert request_id != ""


def test_newline_in_request_id_is_replaced():
    request_id = choose_request_id("bad\nid")
    assert "\n" not in request_id
    assert request_id != "bad\nid"


def test_formatter_drops_passwords_and_tokens():
    formatter = JsonFormatter()
    record = logging.LogRecord("app.request", logging.INFO, __file__, 1, "request", (), None)
    record.request_id = "abc"
    record.path = "/auth/login"
    record.status = 401
    record.latency_ms = 3
    record.password = "at-least-8"
    record.token = "secret-token"
    line = formatter.format(record)
    payload = json.loads(line)
    assert payload["request_id"] == "abc"
    assert payload["message"] == "request"
    assert "at-least-8" not in line
    assert "secret-token" not in line


def test_home_page_has_signup(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Sign up" in response.text
    assert "Ask a question" in response.text
