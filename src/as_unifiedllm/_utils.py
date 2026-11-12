import base64
import gc
import io
import logging
import os
from pathlib import Path
from typing import Any, Union

import requests
from langchain_core.messages import BaseMessage, HumanMessage

from as_unifiedllm._provider import LLMSettings
from as_unifiedllm._model import clean_reasoning_content, get_tokens_for_model, UnifiedChatModel

logger = logging.getLogger("as_unifiedllm")

_IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}


def get_llm(llm_setting: LLMSettings) -> "UnifiedChatModel":  # noqa: F821
    from as_unifiedllm._model import UnifiedChatModel
    return UnifiedChatModel(
        provider=llm_setting.provider,
        model_name=llm_setting.model_name,
        temperature=llm_setting.temperature,
        context_window=llm_setting.context_window,
        top_p=llm_setting.top_p,
        thinking=not llm_setting.no_thinking,
    )


def get_message_content(msg: BaseMessage) -> str:
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        content = " ".join(str(c) for c in msg.content)
    else:
        content = str(msg.content)

    return clean_reasoning_content(content)


def get_token_count(messages: list[BaseMessage]) -> int:
    return sum(len(get_message_content(m)) for m in messages)


def extract_token_usage(response) -> dict:
    if hasattr(response, 'usage_metadata') and isinstance(response.usage_metadata, dict):
        return response.usage_metadata.copy()
    if hasattr(response, 'response_metadata') and isinstance(response.response_metadata, dict):
        if 'token_usage' in response.response_metadata:
            return response.response_metadata['token_usage'].copy()
    return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}


def clear_gpu_memory():
    gc.collect()


def unload_ollama_model(model_name: str):
    base_url = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL")
                or os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
    try:
        logger.info(f"Requesting Ollama to unload model: {model_name}")
        requests.post(f"{base_url}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5)
    except Exception as e:
        logger.warning(f"Failed to unload Ollama model {model_name}: {e}")


def infer_vision(
    settings: LLMSettings,
    image_source: Union[str, Path, "Any"],  # str/Path or PIL.Image
    prompt: str,
) -> tuple[str, dict]:
    """Send an image + prompt to any vision-capable LLM. Returns (text, token_usage)."""
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore

    mime = "image/jpeg"
    if Image and isinstance(image_source, Image.Image):
        buf = io.BytesIO()
        image_source.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
    else:
        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".heic", ".heif"}:
            if Image is None:
                raise ImportError("Pillow required for HEIC conversion")
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            mime = _IMAGE_MIME.get(suffix, "image/jpeg")
            b64 = base64.b64encode(path.read_bytes()).decode()

    llm = get_llm(settings)
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ])
    response = llm.invoke([message])
    return get_message_content(response), extract_token_usage(response)


def get_ollama_context_length(model: str, base_url: str | None = None) -> int | None:
    if base_url is None:
        base_url = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL")
                    or os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
    r = requests.post(f"{base_url}/api/show", json={"name": model})
    r.raise_for_status()
    return r.json().get("model_info", {}).get("llama.context_length")
