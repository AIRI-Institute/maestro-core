from collections import defaultdict
from types import SimpleNamespace
from typing import Annotated, Protocol

from mmar_mapi.services import LLMEndpointMetadata, LLMHubMetadata
from pydantic import AfterValidator, BaseModel, ConfigDict

StrDict = dict[str, str | bool | int | float | dict]


class LLMEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    descriptor: str
    caption: str
    # -1 for disabled
    concurrency_limit: int = -1
    args: StrDict
    extra: StrDict | None = None

    def as_metadata(self) -> LLMEndpointMetadata:
        return LLMEndpointMetadata(key=self.key, caption=self.caption)


def _validate_unique_keys(endpoints: list[LLMEndpointConfig]) -> list[LLMEndpointConfig]:
    endpoints_by_keys = defaultdict(list)
    for epc in endpoints:
        endpoints_by_keys[epc.key].append(epc)
    errors = []
    for key, endpoints_with_key in endpoints_by_keys.items():
        if len(endpoints_with_key) > 1:
            errors.append(f"For key={key} found many endpoints: {endpoints_with_key}")
    if errors:
        raise ValueError("\n".join(errors))
    return endpoints


LLMEndpointConfigs = Annotated[list[LLMEndpointConfig], AfterValidator(_validate_unique_keys)]


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoints: LLMEndpointConfigs

    default_endpoint_key: str = ""
    default_image_endpoint_key: str = ""
    default_file_endpoint_key: str = ""
    # -1 for disabled
    default_concurrency_limit: int = -1

    warmup: bool = False
    wait_seconds_on_llm_retry: list[int | float] | int | float = 1.0

    def as_metadata(self) -> LLMHubMetadata:
        # we can do `return LLMHubMetadata(**self.model_dump())`, but better to pass manually to ensure that no private data passed
        return LLMHubMetadata(
            endpoints=[ep.as_metadata() for ep in self.endpoints],
            default_endpoint_key=self.default_endpoint_key,
        )

    def get_endpoint_config(self, endpoint_key: str) -> LLMEndpointConfig | None:
        return next((epc for epc in self.endpoints if epc.key == endpoint_key), None)


class LLMHubConfig(Protocol):
    llm: LLMConfig


LLM_CONFIG_EMPTY = LLMConfig(endpoints=[])
LLM_HUB_CONFIG_EMPTY = SimpleNamespace(llm=LLM_CONFIG_EMPTY, files_dir=None)
