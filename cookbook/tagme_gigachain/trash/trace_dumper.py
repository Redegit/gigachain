from __future__ import annotations
from typing import Callable, Sequence, Tuple
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.messages import BaseMessage


class TraceDumper(BaseCallbackHandler):
    """Сохраняет все вызовы методов для последующего воспроизведения"""

    def __init__(self, do_dump: bool = True) -> None:
        self.full_trace: List[Tuple[Callable, List[Any], Dict[str, Any]]] = []
        self.do_dump = do_dump

    def _write_dump(self, fc, args_list, kwargs):
        if self.do_dump:
            self.full_trace.append((fc, args_list, kwargs))

    def _reproduce_dump(self):
        for fc, args, kwargs in self.full_trace:
            print(f"Running {fc.__name__} with {len(args)} args, {len(kwargs)} kwargs")
            fc(*args, **kwargs)

    # ---------- LLM (text) ----------
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kw: Any) -> None:
        self._write_dump(self.on_llm_start, [serialized, prompts], kw)

    def on_llm_end(self, response: LLMResult, **kw: Any) -> None:
        self._write_dump(self.on_llm_end, [response], kw)

    def on_llm_error(self, error: BaseException, **kw: Any) -> None:
        self._write_dump(self.on_llm_error, [error], kw)

    # ---------- ChatModel ----------
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kw: Any) -> None:
        self._write_dump(self.on_chat_model_start, [serialized, messages], kw)

    def on_chat_model_end(self, response: LLMResult, **kw: Any) -> None:
        self._write_dump(self.on_chat_model_end, [response], kw)

    def on_chat_model_error(self, error: BaseException, **kw: Any) -> None:
        self._write_dump(self.on_chat_model_error, [error], kw)

    # ---------- Tools ----------
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kw: Any) -> None:
        self._write_dump(self.on_tool_start, [serialized, input_str], kw)

    def on_tool_end(self, output: str, **kw: Any) -> None:
        self._write_dump(self.on_tool_end, [output], kw)

    def on_tool_error(self, error: BaseException, **kw: Any) -> None:
        self._write_dump(self.on_tool_error, [error], kw)

    # ---------- Chains ----------
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kw: Any) -> None:
        self._write_dump(self.on_chain_start, [serialized, inputs], kw)

    def on_chain_end(self, outputs: Dict[str, Any], **kw: Any) -> None:
        self._write_dump(self.on_chain_end, [outputs], kw)

    def on_chain_error(self, error: BaseException, **kw: Any) -> None:
        self._write_dump(self.on_chain_error, [error], kw)

    # ---------- Retrievers (если используете) ----------
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kw: Any) -> None:
        self._write_dump(self.on_retriever_start, [serialized, query], kw)

    def on_retriever_end(self, documents: Sequence[Any], **kw: Any) -> None:
        self._write_dump(self.on_retriever_end, [documents], kw)

    def on_retriever_error(self, error: BaseException, **kw: Any) -> None:
        self._write_dump(self.on_retriever_error, [error], kw)

    # ---------- Text stream helpers (необязательные, но полезны) ----------
    def on_text(self, text: str, **kw: Any) -> None:
        self._write_dump(self.on_text, [text], kw)

    def on_retry(self, retry_state: Any, **kw: Any) -> None:
        self._write_dump(self.on_retry, [retry_state], kw)
