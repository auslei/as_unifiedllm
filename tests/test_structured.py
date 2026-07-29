import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ic_unifiedllm._structured import (
    _coerce_message_text,
    _normalize_schema_echo,
    _recover_schema_value,
    build_structured_prompt,
    parse_structured_output,
    structured_output_repair_fn,
)


class _Sample(BaseModel):
    name: str
    score: int


# --- _coerce_message_text ---

def test_coerce_string():
    assert _coerce_message_text("hello") == "hello"

def test_coerce_list():
    assert _coerce_message_text([{"text": "a"}, {"text": "b"}]) == "ab"

def test_coerce_non_string():
    assert _coerce_message_text(42) == "42"


# --- _recover_schema_value ---

def test_recover_plain_value():
    assert _recover_schema_value("x") == "x"
    assert _recover_schema_value(1) == 1

def test_recover_value_key():
    assert _recover_schema_value({"value": "unwrapped"}) == "unwrapped"

def test_recover_anyof():
    assert _recover_schema_value({"anyOf": [{"value": "first"}, {"value": "second"}]}) == "first"

def test_recover_enum():
    assert _recover_schema_value({"enum": ["a", "b"]}) == "a"

def test_recover_properties():
    result = _recover_schema_value({"properties": {"x": "val"}})
    assert result == {"x": "val"}

def test_recover_list():
    assert _recover_schema_value([{"value": 1}, {"value": 2}]) == [1, 2]

def test_recover_none_for_pure_schema():
    # A dict with only schema keywords and no data returns None
    result = _recover_schema_value({"type": "string", "title": "Foo", "description": "bar"})
    assert result is None


# --- _normalize_schema_echo ---

def test_normalize_schema_echo_direct():
    data = {"name": "Alice", "score": 10}
    result = _normalize_schema_echo(data, _Sample)
    assert result == data

def test_normalize_schema_echo_wrapped_in_properties():
    data = {"properties": {"name": "Bob", "score": 5}}
    result = _normalize_schema_echo(data, _Sample)
    assert result == {"name": "Bob", "score": 5}

def test_normalize_schema_echo_non_dict_passthrough():
    assert _normalize_schema_echo("raw", _Sample) == "raw"


# --- build_structured_prompt ---

def test_build_structured_prompt_contains_schema():
    prompt = build_structured_prompt("Do something", _Sample)
    schema_json = json.dumps(_Sample.model_json_schema(), indent=2)
    assert "Do something" in prompt
    assert schema_json in prompt
    assert "CRITICAL OUTPUT INSTRUCTIONS" in prompt

def test_build_structured_prompt_instructs_no_code_block():
    prompt = build_structured_prompt("x", _Sample)
    assert "markdown code blocks" in prompt


# --- parse_structured_output ---

def test_parse_direct_json():
    raw = json.dumps({"name": "Alice", "score": 7})
    result = parse_structured_output(raw, _Sample)
    assert result.name == "Alice"
    assert result.score == 7

def test_parse_json_with_surrounding_text():
    raw = 'Here is the result: {"name": "Bob", "score": 3} done.'
    result = parse_structured_output(raw, _Sample)
    assert result.name == "Bob"

def test_parse_schema_echo_recovery():
    # LLM echoed the schema instead of filling it
    raw = json.dumps({"properties": {"name": "Carol", "score": 9}})
    result = parse_structured_output(raw, _Sample)
    assert result.name == "Carol"

def test_parse_falls_back_to_llm_repair():
    broken = "name: Dave, score: 2"  # not valid JSON
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"name": "Dave", "score": 2}')
    result = parse_structured_output(broken, _Sample, llm=mock_llm)
    assert result.name == "Dave"
    mock_llm.invoke.assert_called_once()

def test_parse_raises_without_llm_on_garbage():
    with pytest.raises(ValueError, match="_Sample"):
        parse_structured_output("not json at all!", _Sample)


# --- structured_output_repair_fn ---

def test_structured_output_repair_fn():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"fixed": true}')
    result = structured_output_repair_fn(mock_llm, "bad json")
    assert result == '{"fixed": true}'

def test_structured_output_repair_fn_list_content():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=["chunk1", "chunk2"])
    result = structured_output_repair_fn(mock_llm, "bad json")
    assert result == "chunk1"
