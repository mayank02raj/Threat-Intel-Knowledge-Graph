"""Shared test fixtures and configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure API_KEYS is unset so auth is disabled in tests."""
    monkeypatch.delenv("API_KEYS", raising=False)
