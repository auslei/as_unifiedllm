# ic-unifiedllm API Card

Quick reference for using `ic-unifiedllm` in other ic-* services.

## Installation

```toml
# pyproject.toml — add as a path dependency
[tool.uv.sources]
ic-unifiedllm = { path = "../ic_unifiedllm", editable = true }
```

```bash
uv add ic-unifiedllm
```

---

## Core concepts

| Class / function | Purpose |
|-----------------|---------|
| `LLMSettings` | Config dataclass: provider, model, temperature, context window |
| `get_llm(settings)` | Factory — returns a ready `UnifiedChatModel` |
| `UnifiedChatModel` | LangChain `BaseChatModel` subclass; use directly if you need more control |

---

## Basic usage

```python
from ic_unifiedllm import LLMSettings, get_llm

llm = get_llm(LLMSettings(provider="ollama", model_name="qwen3:14b"))

response = llm.invoke("Summarise the French Revolution in one sentence.")
print(response.content)
```

---

## Providers

### Ollama (local)
```python
llm = get_llm(LLMSettings(
    provider="ollama",
    model_name="qwen3:14b",
    context_window=32768,   # optional, default 16384
))
# Env: OLLAMA_BASE_URL (default http://localhost:11434)
```

### OpenAI
```python
llm = get_llm(LLMSettings(provider="openai", model_name="gpt-4o"))
# Env: OPENAI_API_KEY (required)
```

### DeepSeek
```python
llm = get_llm(LLMSettings(provider="deepseek", model_name="deepseek-reasoner"))
# Env: DEEPSEEK_API_KEY (required)
```

### LMStudio (local OpenAI-compat)
```python
llm = get_llm(LLMSettings(provider="lms", model_name="my-local-model"))
# Env: LMSTUDIO_BASE_URL (default http://localhost:1234/v1)
```

### OpenRouter
```python
llm = get_llm(LLMSettings(provider="openrouter", model_name="meta-llama/llama-3-70b-instruct"))
# Env: OPENROUTER_API_KEY (required)
```

### LlamaCpp (direct GGUF)
```python
llm = get_llm(LLMSettings(provider="llama_cpp", model_name="qwen3:14b"))
# Model path resolved from LLAMA_CPP_MODEL_PATHS in _provider.py
```

---

## Structured output

Pass a Pydantic model as `schema=` to get a typed object back. The prompt is
automatically wrapped with JSON instructions; the response is parsed with
fallback LLM repair if needed.

```python
from pydantic import BaseModel
from ic_unifiedllm import LLMSettings, get_llm

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

llm = get_llm(LLMSettings(provider="ollama", model_name="qwen3:14b"))
result: Summary = llm.invoke("Analyse this review: 'Great product, fast shipping!'", schema=Summary)
print(result.title)
print(result.key_points)
```

---

## Thinking / reasoning control

```python
# Disable reasoning traces globally
llm = get_llm(LLMSettings(provider="ollama", model_name="qwen3:14b", no_thinking=True))

# Or override per-call (creates a temporary variant, doesn't mutate the instance)
response = llm.invoke("Quick question", thinking=False)
```

When `thinking=False`:
- `<think>…</think>` and `<tool_call>…</tool_call>` blocks are stripped from responses
- DeepSeek: swaps `deepseek-reasoner` → `deepseek-chat`
- OpenAI: sets `reasoning_effort="low"`
- Ollama: sets `reasoning=False`

---

## Async and streaming

```python
# Async invoke
response = await llm.ainvoke("Hello")

# Sync stream
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

# Async stream
async for chunk in llm.astream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

---

## Tool binding

`UnifiedChatModel` is a LangChain `BaseChatModel`, so standard tool binding works:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}"

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Sydney?")
```

---

## Vision / image input

`infer_vision` sends an image and a text prompt to any vision-capable model and
returns `(text, token_usage)`. The image is encoded to base64 JPEG internally.

```python
from ic_unifiedllm import LLMSettings, infer_vision

settings = LLMSettings(provider="ollama", model_name="llava:13b")

# From a file path (JPEG / PNG / WebP / etc.)
text, usage = infer_vision(settings, "/path/to/photo.jpg", "What is in this image?")

# From a pathlib.Path
from pathlib import Path
text, usage = infer_vision(settings, Path("screenshot.png"), "Describe the UI.")

# From a PIL.Image (requires Pillow)
from PIL import Image
img = Image.open("scan.heic")          # HEIC/HEIF also accepted when Pillow is installed
text, usage = infer_vision(settings, img, "Extract all text from this document.")

print(text)
print(usage)  # {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}
```

**Notes:**
- Pillow (`pip install Pillow`) is optional — only needed for PIL.Image input or HEIC/HEIF files. JPEG/PNG/WebP paths work without it.
- Any provider whose underlying model supports `image_url` content blocks will work (Ollama vision models, OpenAI GPT-4o, OpenRouter vision routes, LMStudio vision models).
- HEIC/HEIF files are converted to JPEG before sending.

---

## Utilities

```python
from ic_unifiedllm import (
    infer_vision,           # image + prompt → (text, token_usage)
    get_message_content,    # extract plain text from any AIMessage
    get_token_count,        # rough char-based count across a message list
    extract_token_usage,    # pull input/output/total from a response object
    unload_ollama_model,    # release a model from Ollama's GPU memory
    get_ollama_context_length,  # query actual context length from Ollama
    clear_gpu_memory,       # gc.collect() — call after unloading llama_cpp models
)

usage = extract_token_usage(response)
# {"input_tokens": 42, "output_tokens": 120, "total_tokens": 162}
```

---

## LLM instance cache

`UnifiedChatModel` caches the underlying provider client by
`(provider, model_name, temperature, context_window, top_p, thinking)`.
Constructing the same settings twice returns the same client — safe to call
`get_llm` on every request without overhead.
