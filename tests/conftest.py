"""Shared pytest fixtures for sgDDG tests."""
import pytest
import sqlite3


@pytest.fixture
def db_connection():
    """Provide an in-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def mock_config():
    """Provide a minimal config dict for testing."""
    return {
        "generation": {"llm_provider": "gemini", "llm_model": "test-model"},
        "api_keys": {"gemini": None},
        "kg": {"vector_dimension": 384},
    }
