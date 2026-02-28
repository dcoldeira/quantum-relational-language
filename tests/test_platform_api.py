"""Tests for the QRL Platform REST API (platform/api.py).

All tests mock the QuantumAILoop and template execution so no real LLM or
QRL computation is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from platform.api import create_app
from platform.executor import ExecutionResult
from platform.templates import TEMPLATES


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture()
def mock_loop():
    """A QuantumAILoop whose ask_full() returns a canned response."""
    loop = MagicMock()
    exec_result = ExecutionResult(
        value={"S": 2.83, "violates_classical": True},
        code="result = chsh_test(trials=10)",
    )
    loop.ask_full.return_value = ("Quantum entanglement confirmed (S=2.83).", exec_result)
    return loop


@pytest.fixture()
def client(mock_loop):
    """TestClient built around a fresh app that uses the mock loop."""
    app = create_app(loop=mock_loop)
    return TestClient(app)


# ------------------------------------------------------------------ #
# /health                                                             #
# ------------------------------------------------------------------ #


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ------------------------------------------------------------------ #
# /ask                                                                #
# ------------------------------------------------------------------ #


def test_ask_returns_200(client, mock_loop):
    r = client.post("/ask", json={"question": "Is this quantum?"})
    assert r.status_code == 200


def test_ask_response_shape(client, mock_loop):
    r = client.post("/ask", json={"question": "Is this quantum?"})
    body = r.json()
    assert "answer" in body
    assert "code" in body
    assert "value" in body
    assert "ok" in body


def test_ask_answer_content(client, mock_loop):
    r = client.post("/ask", json={"question": "Is this quantum?"})
    assert r.json()["answer"] == "Quantum entanglement confirmed (S=2.83)."


def test_ask_ok_flag_true(client, mock_loop):
    r = client.post("/ask", json={"question": "Is this quantum?"})
    assert r.json()["ok"] is True


def test_ask_ok_flag_false_on_error(mock_loop):
    """When ask_full returns an errored ExecutionResult, ok should be False."""
    failed = ExecutionResult(error="NameError: foo", code="result = foo")
    mock_loop.ask_full.return_value = ("Something went wrong.", failed)
    app = create_app(loop=mock_loop)
    c = TestClient(app)
    r = c.post("/ask", json={"question": "broken question"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_ask_calls_ask_full_with_question(client, mock_loop):
    client.post("/ask", json={"question": "What is entanglement?"})
    mock_loop.ask_full.assert_called_once_with("What is entanglement?", verbose=False)


def test_ask_verbose_flag_passed_through(client, mock_loop):
    client.post("/ask", json={"question": "Q", "verbose": True})
    mock_loop.ask_full.assert_called_once_with("Q", verbose=True)


# ------------------------------------------------------------------ #
# /templates                                                          #
# ------------------------------------------------------------------ #


def test_list_templates_returns_200(client):
    r = client.get("/templates")
    assert r.status_code == 200


def test_list_templates_count(client):
    r = client.get("/templates")
    assert len(r.json()) == 5


def test_list_templates_schema(client):
    r = client.get("/templates")
    for item in r.json():
        assert "name" in item
        assert "domain" in item
        assert "question" in item
        assert "description" in item


def test_list_templates_names(client):
    r = client.get("/templates")
    names = {t["name"] for t in r.json()}
    expected = {t.name for t in TEMPLATES}
    assert names == expected


# ------------------------------------------------------------------ #
# /templates/{name}/run                                               #
# ------------------------------------------------------------------ #


def _make_template_result(name: str):
    """Return a mock TemplateResult for the named template."""
    exec_result = ExecutionResult(
        value={"mocked": True},
        code="result = {'mocked': True}",
    )
    tr = MagicMock()
    tr.answer = f"Mocked answer for {name}"
    tr.exec_result = exec_result
    tr.value = exec_result.value
    tr.ok = True
    return tr


@pytest.fixture()
def patched_templates(monkeypatch):
    """Patch every template's run() to return a canned TemplateResult."""
    for t in TEMPLATES:
        tr = _make_template_result(t.name)
        monkeypatch.setattr(t, "run", lambda _tr=tr: _tr)


def test_run_template_not_found(client):
    r = client.post("/templates/nonexistent/run")
    assert r.status_code == 404
    assert "nonexistent" in r.json()["detail"]


def test_run_template_bell(client, patched_templates):
    r = client.post("/templates/Bell Inequality Test/run")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "Bell Inequality Test" in body["answer"]


def test_run_template_response_shape(client, patched_templates):
    r = client.post("/templates/Quantum Network Fidelity/run")
    assert r.status_code == 200
    body = r.json()
    for key in ("answer", "code", "value", "ok"):
        assert key in body


def test_run_template_all_names(client, patched_templates):
    """Every template in TEMPLATES should be runnable via the API."""
    for t in TEMPLATES:
        r = client.post(f"/templates/{t.name}/run")
        assert r.status_code == 200, f"Failed for template: {t.name}"
