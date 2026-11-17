# CLAUDE.md

`as-unifiedllm` (v0.1.0) — shared internal Python library wrapping multiple LLM providers behind one LangChain-compatible interface. Installed as an editable path dependency and consumed by `as_agents` and other as-* services.

Usage examples live in `README.md`; this file is the working map.

## Commands

```bash
uv add <pkg>             # add a dependency (never pip install)
uv sync                  # install/update deps from lockfile
uv run pytest tests/     # run tests
uv run python -c "..."   # one-off scripts inside the venv
```

Set `PYTHONPATH` to the repo root when running scripts outside `uv run`.

## Module map (`src/as_unifiedllm/`)

Everything is exported from `__init__.py`.

| Module | What it does |
|--------|-------------|
| `_provider.py` | `LLMProvider` / `LLMSettings` dataclasses; `LLAMA_CPP_MODEL_PATHS` blob paths |
| `_model.py` | `UnifiedChatModel(BaseChatModel)` — main class; in-process LLM cache (`_RAW_LLM_CACHE`) keyed by `(provider, model_name, temp, ctx, top_p, thinking)` |
| `_utils.py` | `get_llm(LLMSettings)` factory; message content, token counting, GPU/Ollama model unloading, `infer_vision` |
| `_structured.py` | `build_structured_prompt` / `parse_structured_output` / `structured_output_repair_fn` — JSON-schema-constrained output with fallback LLM repair |

### Providers

| Provider string | Underlying class | Key env var(s) |
|----------------|-----------------|----------------|
| `ollama` | `ChatOllama` | `OLLAMA_BASE_URL` / `OLLAMA_URL` / `OLLAMA_API_URL` (default `http://localhost:11434`) |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY` |
| `deepseek` | `ChatDeepSeek` | `DEEPSEEK_API_KEY` |
| `lms` | `ChatOpenAI` (OpenAI-compat) | `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`) |
| `openrouter` | `ChatOpenAI` (OpenAI-compat) | `OPENROUTER_API_KEY` |
| `llama_cpp` | `ChatLlamaCpp` | path from `LLAMA_CPP_MODEL_PATHS` in `_provider.py` |

## Behaviour to know (non-obvious)

- **`thinking` flag**: passed through on `invoke`/`ainvoke`/`stream`/`astream`; overrides the instance setting per-call via `_variant()` (creates a temporary variant, doesn't mutate the instance). `thinking=False` strips `<think>…</think>` and `<tool_call>…</tool_call>` blocks (`clean_reasoning_content`); swaps `deepseek-reasoner`→`deepseek-chat`; sets `reasoning_effort="low"` for OpenAI; sets `reasoning=False` for Ollama.
- **Structured output flow**: `invoke(schema=MyModel)` → `build_structured_prompt` appends JSON-schema instructions → `parse_structured_output` (direct JSON → bracket extraction → optional LLM repair via `structured_output_repair_fn`).
- **LLM cache**: underlying provider client cached by `(provider, model_name, temperature, context_window, top_p, thinking)` — same settings return the same client, so `get_llm` is cheap to call per request.

## Conventions (this repo's stack)

- `uv` for all deps (`uv add`, `uv sync`) — never pip/poetry.
- `snake_case` vars/functions, `PascalCase` classes; type hints on every signature.
- Logging via the named logger `logging.getLogger("as_unifiedllm")` — already established in every module; never `print()`. Log at INFO before major steps.
- Pydantic for structured output; mock provider HTTP calls in tests, not `UnifiedChatModel` itself.
- Keep source files well under 30 KB; split into sub-modules as they grow.
- Black / Ruff are the intended formatter/linter but are not yet configured in `pyproject.toml` — add config when touching the build.

## Adding a new provider

1. Add its branch to `UnifiedChatModel._init_underlying_model` in `_model.py`.
2. Export nothing new — `get_llm` and `UnifiedChatModel` already cover it via the `provider` string.
3. Add a test in `tests/test_provider_init.py`.
