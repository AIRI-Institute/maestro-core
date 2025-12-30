from mmar_mapi.api import LLMRequest, LLMResponseExt

from mmar_llm.llm_endpoint import LLMEndpoint


class DummyEndpoint(LLMEndpoint):
    """for tests"""

    def get_response_ext(self, *, request: LLMRequest) -> LLMResponseExt:
        return LLMResponseExt(text=f"Request: {request}")

    def get_embedding(self, *, prompt: str) -> list[float]:
        return [1, 2, 3]
