from loguru import logger
from mmar_mapi.services import (
    LCP,
    LLMCallProps,
    LLMHubAPI,
    LLMHubMetadata,
    LLMRequest,
    LLMResponseExt,
)

LLMHubMetadata.EMPTY = LLMHubMetadata(endpoints=[], default_endpoint_key="")  # type: ignore[attr-defined]


# todo move later to mmar-llm, useful for testing
class LLMHubDummy(LLMHubAPI):
    def get_metadata(self) -> LLMHubMetadata:
        logger.info("#get_metadata, dummy")
        return LLMHubMetadata.EMPTY  # type: ignore[attr-defined]

    def get_response(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> str:
        return self.get_response_ext(request=request, props=props).text

    def get_response_ext(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> LLMResponseExt:
        logger.info("#get_response_ext, dummy")
        res = f"dummy #get_response_ext, request={request!r}, props={props!r}"
        return LLMResponseExt(text=res)

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        logger.info("#get_embedding, dummy")
        return None
