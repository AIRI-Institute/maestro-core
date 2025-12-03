import importlib
import tiktoken

from mmar_mapi.api import LLMPayload


def count_tokens(sentences: list[str]) -> list[int]:
    encoding = tiktoken.get_encoding("cl100k_base")
    return [len(encoding.encode(sentence)) for sentence in sentences]


# todo specify better type
def dump_messages(payload: LLMPayload) -> list[dict]:
    messages_json = [{"role": msg.role, "content": msg.content} for msg in payload.messages]
    return messages_json


def load_dynamically(object_path: str):
    module_path, class_name = object_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
