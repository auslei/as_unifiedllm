import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from as_unifiedllm._model import (
    STOP_TOKENS,
    UnifiedChatModel,
    clean_reasoning_content,
    get_tokens_for_model,
)


# --- clean_reasoning_content ---

def test_strips_think_block():
    assert clean_reasoning_content("<think>hidden</think>answer") == "answer"

def test_strips_think_block_multiline():
    assert clean_reasoning_content("<think>\nline1\nline2\n</think>final") == "final"

def test_strips_tool_call():
    assert clean_reasoning_content("<tool_call>call()</tool_call>result") == "result"

def test_strips_multiple_blocks():
    raw = "<think>t1</think>mid<think>t2</think>end"
    assert clean_reasoning_content(raw) == "midend"

def test_leaves_clean_string_unchanged():
    assert clean_reasoning_content("plain text") == "plain text"

def test_non_string_passthrough():
    assert clean_reasoning_content(42) == 42

def test_orphan_close_tags_removed():
    assert clean_reasoning_content("text</think>more") == "textmore"


# --- get_tokens_for_model ---

def test_qwen_tokens():
    assert get_tokens_for_model("qwen3:14b") == STOP_TOKENS["qwen"]

def test_llama3_tokens():
    assert get_tokens_for_model("llama3.1:8b") == STOP_TOKENS["llama3"]
    assert get_tokens_for_model("llama-3-8b") == STOP_TOKENS["llama3"]

def test_mistral_tokens():
    assert get_tokens_for_model("mistral:7b") == STOP_TOKENS["mistral"]
    assert get_tokens_for_model("mixtral:8x7b") == STOP_TOKENS["mistral"]

def test_gemma_tokens():
    assert get_tokens_for_model("gemma2:9b") == STOP_TOKENS["gemma"]

def test_phi3_tokens():
    assert get_tokens_for_model("phi3:mini") == STOP_TOKENS["phi3"]

def test_deepseek_llama_tokens():
    assert get_tokens_for_model("deepseek-llama3") == STOP_TOKENS["llama3"]

def test_deepseek_qwen_tokens():
    assert get_tokens_for_model("deepseek-qwen") == STOP_TOKENS["qwen"]

def test_deepseek_base_tokens():
    assert get_tokens_for_model("deepseek-v2") == STOP_TOKENS["deepseek_base"]

def test_unknown_model_default():
    assert get_tokens_for_model("some-unknown-model") == STOP_TOKENS["default"]


# --- UnifiedChatModel (provider calls mocked) ---

def _make_ucm(**kwargs) -> UnifiedChatModel:
    """Build a UnifiedChatModel with a mock underlying model (no provider init)."""
    mock_llm = MagicMock()
    defaults = dict(provider="ollama", model_name="test:7b", model=mock_llm)
    defaults.update(kwargs)
    return UnifiedChatModel(**defaults)


def test_ucm_thinking_true_no_strip():
    ucm = _make_ucm(thinking=True)
    response = AIMessage(content="<think>hidden</think>answer")
    ucm.model.invoke.return_value = response
    result = ucm.invoke("hello")
    assert result.content == "<think>hidden</think>answer"


def test_ucm_thinking_false_strips_content():
    ucm = _make_ucm(thinking=False)
    response = AIMessage(content="<think>hidden</think>answer")
    ucm.model.invoke.return_value = response
    result = ucm.invoke("hello")
    assert result.content == "answer"


def test_ucm_invoke_with_schema():
    class _Out(BaseModel):
        value: int

    ucm = _make_ucm(thinking=False)
    ucm.model.invoke.return_value = AIMessage(content='{"value": 42}')
    result = ucm.invoke("give me a number", schema=_Out)
    assert isinstance(result, _Out)
    assert result.value == 42


def test_ucm_invoke_thinking_override_uses_variant():
    ucm = _make_ucm(thinking=True)
    response = AIMessage(content="<think>x</think>clean")
    # The variant will be a fresh UnifiedChatModel; patch _init_underlying_model
    with patch.object(UnifiedChatModel, "_init_underlying_model", return_value=MagicMock(invoke=MagicMock(return_value=response))):
        result = ucm.invoke("hi", thinking=False)
    assert result.content == "clean"


def test_ucm_variant_thinking_flag():
    ucm = _make_ucm(thinking=True)
    with patch.object(UnifiedChatModel, "_init_underlying_model", return_value=MagicMock()):
        variant = ucm._variant(thinking=False)
    assert variant.thinking is False
    assert variant.provider == ucm.provider
    assert variant.model_name == ucm.model_name


def test_ucm_inject_schema_wraps_string():
    class _Out(BaseModel):
        x: str

    ucm = _make_ucm()
    result = ucm._inject_schema("my prompt", _Out)
    assert "my prompt" in result
    assert "CRITICAL OUTPUT INSTRUCTIONS" in result


def test_ucm_inject_schema_wraps_last_human_message():
    class _Out(BaseModel):
        x: str

    ucm = _make_ucm()
    messages = [HumanMessage(content="ask")]
    result = ucm._inject_schema(messages, _Out)
    assert isinstance(result[-1], HumanMessage)
    assert "CRITICAL OUTPUT INSTRUCTIONS" in result[-1].content


def test_ucm_bind_tools_returns_ucm():
    ucm = _make_ucm()
    ucm.model.bind_tools.return_value = MagicMock()
    bound = ucm.bind_tools([])
    assert isinstance(bound, UnifiedChatModel)


def test_ucm_bind_returns_ucm():
    ucm = _make_ucm()
    ucm.model.bind.return_value = MagicMock()
    bound = ucm.bind(stop=["<end>"])
    assert isinstance(bound, UnifiedChatModel)


@pytest.mark.asyncio
async def test_ucm_ainvoke_strips_reasoning():
    ucm = _make_ucm(thinking=False)
    ucm.model.ainvoke = AsyncMock(return_value=AIMessage(content="<think>skip</think>result"))
    result = await ucm.ainvoke("prompt")
    assert result.content == "result"
