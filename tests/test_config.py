"""config tests: key priority, empty-value handling, redaction. No key needed."""

import importlib

import pytest

import config
from config import MissingKeyError, redact, resolve_llm_settings


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GPT_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_no_key_raises_instead_of_using_a_fake_default():
    with pytest.raises(MissingKeyError):
        resolve_llm_settings()


def test_empty_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(MissingKeyError):
        resolve_llm_settings()


def test_openrouter_key_is_preferred_and_models_get_the_provider_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = resolve_llm_settings()
    assert s.api_key == "sk-or-test"
    assert s.base_url == config.OPENROUTER_BASE_URL
    assert s.model == config._prefixed(config.MODEL)
    assert s.embedding_model == "openai/text-embedding-ada-002"


def test_openai_key_is_used_without_a_prefix_or_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = resolve_llm_settings()
    assert (s.base_url, s.model, s.embedding_model) == (None, config.MODEL, "text-embedding-ada-002")


def test_old_gpt_key_name_still_works(monkeypatch):
    monkeypatch.setenv("GPT_KEY", "sk-old")
    assert resolve_llm_settings().api_key == "sk-old"


def test_key_pasted_in_the_ui_wins_and_its_prefix_picks_the_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert resolve_llm_settings("sk-or-pasted").base_url == config.OPENROUTER_BASE_URL
    s = resolve_llm_settings("  sk-pasted  ")
    assert (s.api_key, s.base_url) == ("sk-pasted", None)


def test_empty_llm_model_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    # Keep the reload from re-reading a real .env into the process environment.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    try:
        importlib.reload(config)
        assert config.MODEL == "gpt-6-luna"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_redact_hides_the_key_and_anything_shaped_like_one():
    text = "Incorrect API key provided: sk-abc123456789. Key sk-or-v1-secretvalue failed"
    out = redact(text, "sk-or-v1-secretvalue")
    assert "secretvalue" not in out and "abc123456789" not in out
    assert redact("nothing secret here") == "nothing secret here"


def test_settings_repr_never_contains_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-appear")
    assert "should-not-appear" not in repr(resolve_llm_settings())
