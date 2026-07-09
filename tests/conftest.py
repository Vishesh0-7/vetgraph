"""Shared pytest fixtures for VetGraph tests."""

from unittest.mock import Mock

import instructor
import pytest


@pytest.fixture
def mock_openai_client():
    """Compatibility fixture for legacy tests that still expect an OpenAI-style client."""
    mock_client = Mock(spec=instructor.Instructor)
    mock_client.beta = Mock()
    mock_client.beta.chat = Mock()
    mock_client.beta.chat.completions = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    return mock_client


@pytest.fixture(autouse=True)
def patch_legacy_openai_constructor(monkeypatch, mock_openai_client):
    """Route legacy api_key-based VetGraph tests to the mock client instead of the real SDK."""
    import vetgraph.core as core

    monkeypatch.setattr(core, "OpenAI", lambda *args, **kwargs: mock_openai_client)
