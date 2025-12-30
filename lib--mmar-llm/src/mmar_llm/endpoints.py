from mmar_llm.llm_endpoint import LLMEndpoint
from mmar_llm.airi_endpoint import AiriChatEndpoint
from mmar_llm.dummy_endpoint import DummyEndpoint
from mmar_llm.fusion_brain_endpoint import FusionBrainEndpoint
from mmar_llm.gigachat_endpoint import (
    GigaChatEndpoint,
    GigaChatSberdevicesEndpoint,
)
from mmar_llm.openrouter_endpoint import OpenRouterEndpoint
from mmar_llm.utils import load_dynamically
from mmar_llm.yandex_gpt_endpoint import YandexGPTEndpoint

ENDPOINTS = [
    AiriChatEndpoint,
    DummyEndpoint,
    FusionBrainEndpoint,
    GigaChatEndpoint,
    GigaChatSberdevicesEndpoint,
    OpenRouterEndpoint,
    YandexGPTEndpoint,
]
ALIASES = [getattr(ep, "ALIAS", None) for ep in ENDPOINTS]
ALIASES_MAP = {alias: ep for alias, ep in zip(ALIASES, ENDPOINTS) if alias}


def find_llm_endpoint(descriptor: str) -> type[LLMEndpoint] | None:
    try:
        return load_dynamically(descriptor)
    except Exception:
        pass

    ep = ALIASES_MAP.get(descriptor)
    if ep:
        return ep

    return None
