"""Deployment-wrapper checks that do not contact the Modal control plane."""

from __future__ import annotations

import pytest


def test_modal_web_app_mounts_the_production_demo() -> None:
    pytest.importorskip("modal")
    from fastapi.testclient import TestClient

    from modal_app import APP_NAME, create_web_app

    assert APP_NAME == "phone-accentedness-scorer"
    with TestClient(create_web_app()) as client:
        health = client.get("/healthz")
        root = client.get("/")
        config = client.get("/config")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert root.status_code == 200
    assert "Phoneme Accentedness Scorer" in root.text
    assert config.status_code == 200
    assert config.json()["enable_queue"] is True
