from ic_unifiedllm._provider import LLAMA_CPP_MODEL_PATHS, LLMProvider, LLMSettings


def test_llm_provider_fields():
    p = LLMProvider(provider="ollama", model_name="qwen3:8b")
    assert p.provider == "ollama"
    assert p.model_name == "qwen3:8b"

def test_llm_provider_getitem():
    p = LLMProvider(provider="openai", model_name="gpt-4o")
    assert p["provider"] == "openai"
    assert p["model_name"] == "gpt-4o"

def test_llm_settings_defaults():
    s = LLMSettings(provider="ollama", model_name="qwen3:8b")
    assert s.temperature == 0.0
    assert s.top_p == 1.0
    assert s.no_thinking is False
    assert s.context_window is None

def test_llm_settings_override():
    s = LLMSettings(provider="deepseek", model_name="deepseek-reasoner", temperature=0.7, no_thinking=True)
    assert s.temperature == 0.7
    assert s.no_thinking is True

def test_llm_settings_getitem():
    s = LLMSettings(provider="openai", model_name="o1", temperature=0.5)
    assert s["temperature"] == 0.5

def test_llama_cpp_model_paths_defined():
    assert "qwen3:14b" in LLAMA_CPP_MODEL_PATHS
    assert "qwen3:8b" in LLAMA_CPP_MODEL_PATHS
    for path in LLAMA_CPP_MODEL_PATHS.values():
        assert path.startswith("/")
