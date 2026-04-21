"""Legacy API wrapper for OpenAI client compatibility."""

from mmar_mapi.services.llm_hub import (
    LCP,
    LLMCallProps,
    LLMEndpointMetadata,
    LLMHubAPI,
    LLMHubMetadata,
    LLMPayload,
    LLMRequest,
    LLMResponseExt,
)
from openai import OpenAI


class LLMHubOpenAIWrapper(LLMHubAPI):
    """LLMHubAPI implementation wrapping OpenAI client."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def get_metadata(self) -> LLMHubMetadata:
        models = self._client.models.list()
        endpoints = [LLMEndpointMetadata(key=m.id, caption=m.id) for m in models.data]
        default = next((m["id"] for m in models.model_dump().get("data", []) if m.get("default")), "")
        return LLMHubMetadata(endpoints=endpoints, default_endpoint_key=default)

    def get_response(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> str:
        payload = LLMPayload.parse(request)
        messages = [msg.model_dump() for msg in payload.messages]
        completions = self._client.chat.completions.create(model=props.endpoint_key, messages=messages)
        return completions.choices[0].message.content or ""

    def get_response_ext(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> LLMResponseExt:
        return LLMResponseExt(text=self.get_response(request=request, props=props))

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        return self._client.embeddings.create(model=props.endpoint_key, input=prompt).data[0].embedding


def as_llm_hub(oclient: OpenAI) -> LLMHubAPI:
    return LLMHubOpenAIWrapper(oclient)


def fix_base_url(base_url: str):
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    if not base_url.startswith("http"):
        base_url = "http://" + base_url

    return base_url


def as_llm_hub_for_openai(base_url: str, api_key: str = "") -> LLMHubAPI:
    base_url = fix_base_url(base_url)
    oclient = OpenAI(base_url=base_url, api_key=api_key)
    return LLMHubOpenAIWrapper(oclient)
