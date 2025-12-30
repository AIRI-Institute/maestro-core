from mmar_mapi.services import LLMRequest, LLMResponseExt


class LLMEndpoint:
    _key: str | None

    def get_response_ext(self, *, request: LLMRequest) -> LLMResponseExt:
        raise NotImplementedError

    def get_embedding(self, *, prompt: str) -> list[float]:
        raise NotImplementedError

    # helpers

    def get_response(self, *, request: LLMRequest) -> str:
        response_ext = self.get_response_ext(request=request)
        return response_ext.text


class LLMEndpointFacade(LLMEndpoint):
    def __init__(self, base: LLMEndpoint, decorator):
        self._base = base

        self._get_response = decorator(self._base.get_response)
        self._get_response_ext = decorator(self._base.get_response_ext)
        self._get_embedding = decorator(self._base.get_embedding)

    def get_response(self, *, request: LLMRequest) -> str:
        return self._get_response(request=request)

    def get_response_ext(self, *, request: LLMRequest) -> LLMResponseExt:
        return self._get_response_ext(request=request)

    def get_embedding(self, *, prompt: str) -> list[float]:
        return self._get_embedding(prompt=prompt)
