from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from as_unifiedllm._provider import LLMSettings
from as_unifiedllm._utils import (
    extract_token_usage,
    get_message_content,
    get_ollama_context_length,
    get_token_count,
    unload_ollama_model,
)


# --- get_message_content ---

def test_plain_string_content():
    msg = AIMessage(content="hello world")
    assert get_message_content(msg) == "hello world"

def test_strips_think_tags():
    msg = AIMessage(content="<think>skip</think>answer")
    assert get_message_content(msg) == "answer"

def test_list_content_joined():
    msg = AIMessage(content=["part1", "part2"])
    assert "part1" in get_message_content(msg)
    assert "part2" in get_message_content(msg)


# --- get_token_count ---

def test_token_count_empty():
    assert get_token_count([]) == 0

def test_token_count_sums_lengths():
    msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
    assert get_token_count(msgs) == len("hello") + len("world")


# --- extract_token_usage ---

def test_extract_from_usage_metadata():
    response = MagicMock()
    response.usage_metadata = {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
    result = extract_token_usage(response)
    assert result == {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}

def test_extract_from_response_metadata():
    response = MagicMock(spec=[])  # no usage_metadata attr
    response.response_metadata = {"token_usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}}
    result = extract_token_usage(response)
    assert result["input_tokens"] == 3

def test_extract_returns_zeros_on_missing():
    response = MagicMock(spec=[])  # no relevant attrs
    result = extract_token_usage(response)
    assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def test_extract_returns_copy():
    response = MagicMock()
    original = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    response.usage_metadata = original
    result = extract_token_usage(response)
    result["input_tokens"] = 999
    assert original["input_tokens"] == 1


# --- unload_ollama_model ---

def test_unload_ollama_model_posts_correct_payload():
    with patch("as_unifiedllm._utils.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        unload_ollama_model("qwen3:14b")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {"model": "qwen3:14b", "keep_alive": 0}

def test_unload_ollama_model_swallows_errors():
    with patch("as_unifiedllm._utils.requests.post", side_effect=Exception("conn refused")):
        unload_ollama_model("qwen3:14b")  # must not raise


# --- get_ollama_context_length ---

def test_get_ollama_context_length():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"model_info": {"llama.context_length": 8192}}
    with patch("as_unifiedllm._utils.requests.post", return_value=mock_resp):
        result = get_ollama_context_length("qwen3:14b")
    assert result == 8192

def test_get_ollama_context_length_missing_key():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    with patch("as_unifiedllm._utils.requests.post", return_value=mock_resp):
        result = get_ollama_context_length("unknown-model")
    assert result is None


# --- get_llm (factory) ---

def test_get_llm_returns_unified_chat_model():
    from as_unifiedllm._model import UnifiedChatModel
    from as_unifiedllm._utils import get_llm

    settings = LLMSettings(provider="ollama", model_name="qwen3:8b", no_thinking=True)
    with patch.object(UnifiedChatModel, "_init_underlying_model", return_value=MagicMock()):
        llm = get_llm(settings)
    assert isinstance(llm, UnifiedChatModel)
    assert llm.provider == "ollama"
    assert llm.thinking is False
