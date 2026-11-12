"""Tests for UnifiedChatModel._init_underlying_model — one test per provider branch."""
import os
from unittest.mock import MagicMock, patch

import pytest

from as_unifiedllm._model import UnifiedChatModel, _RAW_LLM_CACHE


def _init(provider, model_name="m", thinking=True, **kwargs):
    """Call _init_underlying_model with cache cleared so each test is isolated."""
    _RAW_LLM_CACHE.clear()
    return UnifiedChatModel._init_underlying_model(
        provider, model_name, temperature=0.0,
        context_window=None, top_p=1.0, thinking=thinking, **kwargs
    )


# --- ollama ---

def test_ollama_default_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    with patch("as_unifiedllm._model.ChatOllama") as MockOllama:
        _init("ollama", "qwen3:8b")
        call_kwargs = MockOllama.call_args[1]
    assert call_kwargs["base_url"] == "http://localhost:11434"

def test_ollama_custom_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    with patch("as_unifiedllm._model.ChatOllama") as MockOllama:
        _init("ollama", "qwen3:8b")
        call_kwargs = MockOllama.call_args[1]
    assert call_kwargs["base_url"] == "http://gpu-box:11434"

def test_ollama_thinking_false_sets_reasoning(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    with patch("as_unifiedllm._model.ChatOllama") as MockOllama:
        _init("ollama", "qwen3:8b", thinking=False)
        call_kwargs = MockOllama.call_args[1]
    assert call_kwargs.get("reasoning") is False

def test_ollama_thinking_true_no_reasoning_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    with patch("as_unifiedllm._model.ChatOllama") as MockOllama:
        _init("ollama", "qwen3:8b", thinking=True)
        call_kwargs = MockOllama.call_args[1]
    assert "reasoning" not in call_kwargs

def test_ollama_uses_default_context_window(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    with patch("as_unifiedllm._model.ChatOllama") as MockOllama:
        _init("ollama", "qwen3:8b")
        call_kwargs = MockOllama.call_args[1]
    assert call_kwargs["num_ctx"] == 16384


# --- openai ---

def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _init("openai", "gpt-4o")

def test_openai_thinking_false_sets_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("as_unifiedllm._model.ChatOpenAI") as MockOpenAI:
        _init("openai", "o1", thinking=False)
        call_kwargs = MockOpenAI.call_args[1]
    assert call_kwargs.get("reasoning_effort") == "low"

def test_openai_thinking_true_no_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("as_unifiedllm._model.ChatOpenAI") as MockOpenAI:
        _init("openai", "gpt-4o", thinking=True)
        call_kwargs = MockOpenAI.call_args[1]
    assert "reasoning_effort" not in call_kwargs


# --- deepseek ---

def test_deepseek_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _init("deepseek", "deepseek-chat")

def test_deepseek_thinking_false_swaps_reasoner(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    with patch("as_unifiedllm._model.ChatDeepSeek") as MockDS:
        _init("deepseek", "deepseek-reasoner", thinking=False)
        call_kwargs = MockDS.call_args[1]
    assert call_kwargs["model"] == "deepseek-chat"

def test_deepseek_thinking_true_keeps_reasoner(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    with patch("as_unifiedllm._model.ChatDeepSeek") as MockDS:
        _init("deepseek", "deepseek-reasoner", thinking=True)
        call_kwargs = MockDS.call_args[1]
    assert call_kwargs["model"] == "deepseek-reasoner"

def test_deepseek_non_reasoner_model_unchanged(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    with patch("as_unifiedllm._model.ChatDeepSeek") as MockDS:
        _init("deepseek", "deepseek-chat", thinking=False)
        call_kwargs = MockDS.call_args[1]
    assert call_kwargs["model"] == "deepseek-chat"


# --- lms (LMStudio) ---

def test_lms_default_base_url(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    with patch("as_unifiedllm._model.ChatOpenAI") as MockOpenAI:
        _init("lms", "local-model")
        call_kwargs = MockOpenAI.call_args[1]
    assert call_kwargs["base_url"] == "http://localhost:1234/v1"
    assert call_kwargs["api_key"].get_secret_value() == "lm-studio"

def test_lms_custom_base_url(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://192.168.1.10:1234/v1")
    with patch("as_unifiedllm._model.ChatOpenAI") as MockOpenAI:
        _init("lms", "local-model")
        call_kwargs = MockOpenAI.call_args[1]
    assert call_kwargs["base_url"] == "http://192.168.1.10:1234/v1"


# --- openrouter ---

def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _init("openrouter", "some/model")

def test_openrouter_uses_correct_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("as_unifiedllm._model.ChatOpenAI") as MockOpenAI:
        _init("openrouter", "some/model")
        call_kwargs = MockOpenAI.call_args[1]
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


# --- llama_cpp ---

def test_llama_cpp_known_model():
    with patch("as_unifiedllm._model.ChatLlamaCpp") as MockCpp:
        _init("llama_cpp", "qwen3:14b")
    MockCpp.assert_called_once()

def test_llama_cpp_unknown_model_raises():
    with pytest.raises(ValueError, match="Model path not defined"):
        _init("llama_cpp", "nonexistent-model")


# --- unknown provider ---

def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        _init("banana", "some-model")


# --- cache ---

def test_same_args_returns_cached_instance(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    with patch("as_unifiedllm._model.ChatOllama") as MockOllama:
        r1 = _init("ollama", "qwen3:8b")
        r2 = UnifiedChatModel._init_underlying_model(
            "ollama", "qwen3:8b", 0.0, None, 1.0, True
        )
    assert r1 is r2
    assert MockOllama.call_count == 1
