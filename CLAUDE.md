# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ic-unifiedllm` is a shared internal Python library (installed as an editable package via uv) that wraps multiple LLM providers behind a single LangChain-compatible interface. It is consumed by `ic_agents` and other ic-* services.

## Commands

```bash
uv add <pkg>             # add a dependency (never pip install)
uv sync                  # install/update deps from lockfile
uv run python -c "..."   # run one-off scripts inside the venv
uv run pytest tests/     # run tests (tests/ dir not yet created)
```

Set `PYTHONPATH` to the repo root when running scripts outside `uv run`.

## Architecture

Four modules, all exported from `__init__.py`:

| Module | What it does |
|--------|-------------|
| `_provider.py` | `LLMProvider` / `LLMSettings` dataclasses; `LLAMA_CPP_MODEL_PATHS` blob paths |
| `_model.py` | `UnifiedChatModel(BaseChatModel)` — the main class; in-process LLM cache (`_RAW_LLM_CACHE`) keyed by `(provider, model_name, temp, ctx, top_p, thinking)` |
| `_utils.py` | `get_llm(LLMSettings)` factory; helpers for message content, token counting, GPU/Ollama model unloading |
| `_structured.py` | `build_structured_prompt` / `parse_structured_output` / `structured_output_repair_fn` — JSON-schema-constrained output with fallback LLM repair |

### Supported providers

| Provider string | Underlying class | Key env var(s) |
|----------------|-----------------|----------------|
| `ollama` | `ChatOllama` | `OLLAMA_BASE_URL` / `OLLAMA_URL` / `OLLAMA_API_URL` |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY` |
| `deepseek` | `ChatDeepSeek` | `DEEPSEEK_API_KEY` |
| `lms` | `ChatOpenAI` (OpenAI-compat) | `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`) |
| `openrouter` | `ChatOpenAI` (OpenAI-compat) | `OPENROUTER_API_KEY` |
| `llama_cpp` | `ChatLlamaCpp` | path from `LLAMA_CPP_MODEL_PATHS` |

### `thinking` flag behaviour

- Passed through on `invoke`/`ainvoke`/`stream`/`astream`; overrides the instance setting per-call via `_variant()`.
- `thinking=False`: strips `<think>…</think>` and `<tool_call>…</tool_call>` blocks from responses (`clean_reasoning_content`). For `deepseek-reasoner` it swaps the model to `deepseek-chat`; for OpenAI it sets `reasoning_effort="low"`; for Ollama it sets `reasoning=False`.

### Structured output flow

`invoke(schema=MyModel)` → `build_structured_prompt` appends JSON schema instructions → response parsed by `parse_structured_output` (direct JSON → bracket extraction → optional LLM repair via `structured_output_repair_fn`).

## Adding a new provider

1. Add its branch to `UnifiedChatModel._init_underlying_model` in `_model.py`.
2. Export nothing new — `get_llm` and `UnifiedChatModel` already cover it via the `provider` string.

## Standards (from `ic_common/GLOBAL_STANDARDS.md`)

### Code style
- `snake_case` for variables and functions; `PascalCase` for classes.
- Type hints on all function signatures.
- Format with **Black** / lint with **Ruff** (not yet configured in `pyproject.toml` — add when touching the build config).
- Imports: alphabetised, minimal.
- Comments only where intent is non-obvious (no narrating what the code does).

### Logging
- Use the named logger `logging.getLogger("ic_unifiedllm")` — already established in every module.
- No bare `print()`. Log at `INFO` before major steps to aid debugging.
- Raise clear, actionable errors; don't let raw tracebacks surface to callers.

### Design
- Prefer straightforward logic over extra wrappers or abstractions.
- Functions should be deterministic and side-effect-free where possible.
- Keep individual source files well under 30 KB; split into sub-modules if they grow.

### Testing
- Tests go in `tests/` at the repo root; manual/integration tests in `tests/manual/`.
- Add or update a test for every behavioural change.
- Prefer small, deterministic unit tests; mock provider HTTP calls, not `UnifiedChatModel` itself.
