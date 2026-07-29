import logging
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

from ic_unifiedllm._provider import LLAMA_CPP_MODEL_PATHS
from ic_unifiedllm._structured import build_structured_prompt, parse_structured_output

logger = logging.getLogger("ic_unifiedllm")

_DEFAULT_LLM_CONTEXT_WINDOW = 16384
_RAW_LLM_CACHE: Dict[tuple, Any] = {}

STOP_TOKENS = {
    "qwen": ["<|im_end|>", "<|endoftext|>"],
    "llama3": ["<|eot_id|>", "<|end_of_text|>"],
    "mistral": ["</s>"],
    "gemma": ["<end_of_turn>", "<eos>"],
    "phi3": ["<|end|>", "<|endoftext|>"],
    "deepseek_base": ["<|EOT|>", "<|end_of_sentence|>"],
    "default": [],
}


def get_tokens_for_model(model_name_or_path: str) -> list:
    s = model_name_or_path.lower()
    if "qwen" in s: return STOP_TOKENS["qwen"]
    if "llama-3" in s or "llama3" in s: return STOP_TOKENS["llama3"]
    if "mistral" in s or "mixtral" in s: return STOP_TOKENS["mistral"]
    if "gemma" in s: return STOP_TOKENS["gemma"]
    if "phi-3" in s or "phi3" in s: return STOP_TOKENS["phi3"]
    if "deepseek" in s:
        if "llama" in s: return STOP_TOKENS["llama3"]
        if "qwen" in s: return STOP_TOKENS["qwen"]
        return STOP_TOKENS["deepseek_base"]
    return STOP_TOKENS["default"]


def clean_reasoning_content(content: str) -> str:
    if not isinstance(content, str):
        return content
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
    content = content.replace('</think>', '').replace('</tool_call>', '')
    return content.strip()


class UnifiedChatModel(BaseChatModel):
    """Unified LangChain wrapper supporting Ollama, OpenAI, DeepSeek, LMStudio, OpenRouter, LlamaCpp."""

    provider: str
    model_name: str
    temperature: float = 0.0
    context_window: Optional[int] = None
    top_p: Optional[float] = 1.0
    thinking: bool = True

    model: Any = Field(exclude=True, default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.model is None:
            start = time.perf_counter()
            self.model = self._init_underlying_model(
                self.provider, self.model_name, self.temperature,
                self.context_window, self.top_p, self.thinking,
            )
            logger.debug(f"Underlying LLM {self.provider}/{self.model_name} ready in {time.perf_counter() - start:.3f}s")

    @staticmethod
    def _init_underlying_model(
        provider: str, model_name: str, temperature: float,
        context_window: Optional[int], top_p: Optional[float], thinking: bool,
    ) -> Any:
        cache_key = (provider, model_name, temperature, context_window, top_p, thinking)
        if cache_key in _RAW_LLM_CACHE:
            return _RAW_LLM_CACHE[cache_key]

        logger.info(f"Initializing NEW raw LLM: {provider}/{model_name} (thinking={thinking})")
        temp = 0.0 if temperature is None else temperature

        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            effective_model = model_name
            if not thinking and "reasoner" in model_name:
                logger.info("thinking=False: Swapping deepseek-reasoner for deepseek-chat")
                effective_model = "deepseek-chat"
            llm = ChatDeepSeek(model=effective_model, api_key=SecretStr(api_key), temperature=temp)

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            kwargs: Dict[str, Any] = {"model": model_name, "api_key": SecretStr(api_key), "temperature": temp}
            if not thinking:
                kwargs["reasoning_effort"] = "low"
            llm = ChatOpenAI(**kwargs)

        elif provider == "lms":
            base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            llm = ChatOpenAI(model=model_name, api_key=SecretStr("lm-studio"), base_url=base_url, temperature=temp)

        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            llm = ChatOpenAI(model=model_name, api_key=SecretStr(api_key),
                             base_url="https://openrouter.ai/api/v1", temperature=temp)

        elif provider == "ollama":
            base_url = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL")
                        or os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
            logger.info(f"Using Ollama base URL: {base_url}")
            kwargs = {
                "model": model_name, "temperature": temp, "top_p": top_p,
                "num_ctx": context_window if context_window is not None else _DEFAULT_LLM_CONTEXT_WINDOW,
                "base_url": base_url,
            }
            if not thinking:
                kwargs["reasoning"] = False
            llm = ChatOllama(**kwargs)

        elif provider == "llama_cpp":
            model_path = LLAMA_CPP_MODEL_PATHS.get(model_name)
            if not model_path:
                raise ValueError(f"Model path not defined for LlamaCpp model: {model_name}")
            llm = ChatLlamaCpp(
                model_path=model_path, temperature=temp, n_gpu_layers=-1,
                max_tokens=8192,
                n_ctx=context_window if context_window is not None else _DEFAULT_LLM_CONTEXT_WINDOW,
                stop=get_tokens_for_model(model_name), verbose=False,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        _RAW_LLM_CACHE[cache_key] = llm
        return llm

    @property
    def _llm_type(self) -> str:
        return f"unified_{self.provider}"

    def _strip_reasoning(self, result: ChatResult) -> ChatResult:
        if not self.thinking:
            for gen in result.generations:
                if hasattr(gen.message, "content") and isinstance(gen.message.content, str):
                    gen.message.content = clean_reasoning_content(gen.message.content)
        return result

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        start = time.perf_counter()
        if hasattr(self.model, "_generate"):
            result = self.model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        else:
            response = self.model.invoke(messages, stop=stop, **kwargs)
            result = ChatResult(generations=[ChatGeneration(message=response)])
        logger.debug(f"Inference completed in {time.perf_counter() - start:.3f}s")
        return self._strip_reasoning(result)

    async def _agenerate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        start = time.perf_counter()
        if hasattr(self.model, "_agenerate"):
            result = await self.model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        else:
            response = await self.model.ainvoke(messages, stop=stop, **kwargs)
            result = ChatResult(generations=[ChatGeneration(message=response)])
        logger.debug(f"Async inference completed in {time.perf_counter() - start:.3f}s")
        return self._strip_reasoning(result)

    def invoke(self, input: Any, config=None, *, schema: Optional[Type[BaseModel]] = None,
               thinking: Optional[bool] = None, **kwargs) -> Union[BaseMessage, Any]:
        if thinking is not None and thinking != self.thinking:
            return self._variant(thinking).invoke(input, config=config, schema=schema, **kwargs)

        if schema:
            input = self._inject_schema(input, schema)

        start = time.perf_counter()
        response = self.model.invoke(input, config=config, **kwargs)
        logger.info(f"Inference completed in {time.perf_counter() - start:.3f}s ({self.provider}/{self.model_name})")

        if not self.thinking and hasattr(response, "content") and isinstance(response.content, str):
            response.content = clean_reasoning_content(response.content)

        if schema:
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
            return parse_structured_output(content, schema, llm=self.model)

        return response

    async def ainvoke(self, input: Any, config=None, *, schema: Optional[Type[BaseModel]] = None,
                      thinking: Optional[bool] = None, **kwargs) -> Union[BaseMessage, Any]:
        if thinking is not None and thinking != self.thinking:
            return await self._variant(thinking).ainvoke(input, config=config, schema=schema, **kwargs)

        if schema:
            input = self._inject_schema(input, schema)

        start = time.perf_counter()
        response = await self.model.ainvoke(input, config=config, **kwargs)
        logger.info(f"Async inference completed in {time.perf_counter() - start:.3f}s ({self.provider}/{self.model_name})")

        if not self.thinking and hasattr(response, "content") and isinstance(response.content, str):
            response.content = clean_reasoning_content(response.content)

        if schema:
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
            return parse_structured_output(content, schema, llm=self.model)

        return response

    def stream(self, input: Any, config=None, *, thinking: Optional[bool] = None, **kwargs) -> Iterator[BaseMessage]:
        if thinking is not None and thinking != self.thinking:
            yield from self._variant(thinking).stream(input, config=config, **kwargs)
            return
        yield from self.model.stream(input, config=config, **kwargs)

    async def astream(self, input: Any, config=None, *, thinking: Optional[bool] = None, **kwargs):
        if thinking is not None and thinking != self.thinking:
            async for chunk in self._variant(thinking).astream(input, config=config, **kwargs):
                yield chunk
            return
        async for chunk in self.model.astream(input, config=config, **kwargs):
            yield chunk

    def bind_tools(self, tools: List[Any], **kwargs) -> "UnifiedChatModel":
        return UnifiedChatModel(
            provider=self.provider, model_name=self.model_name, temperature=self.temperature,
            context_window=self.context_window, top_p=self.top_p, thinking=self.thinking,
            model=self.model.bind_tools(tools, **kwargs),
        )

    def bind(self, **kwargs) -> "UnifiedChatModel":
        return UnifiedChatModel(
            provider=self.provider, model_name=self.model_name, temperature=self.temperature,
            context_window=self.context_window, top_p=self.top_p, thinking=self.thinking,
            model=self.model.bind(**kwargs),
        )

    def _variant(self, thinking: bool) -> "UnifiedChatModel":
        return UnifiedChatModel(
            provider=self.provider, model_name=self.model_name, temperature=self.temperature,
            context_window=self.context_window, top_p=self.top_p, thinking=thinking,
        )

    def _inject_schema(self, input: Any, schema: Type[BaseModel]) -> Any:
        if isinstance(input, str):
            return build_structured_prompt(input, schema)
        if isinstance(input, list) and input:
            last = input[-1]
            if isinstance(last, (HumanMessage, SystemMessage)) and isinstance(last.content, str):
                input = list(input)
                input[-1] = type(last)(content=build_structured_prompt(last.content, schema),
                                       **last.additional_kwargs)
        return input
