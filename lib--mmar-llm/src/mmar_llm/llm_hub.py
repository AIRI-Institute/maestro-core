import mimetypes
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from threading import Lock

from loguru import logger
from mmar_mapi import FileStorage
from mmar_mapi.services import (
    LCP,
    RESPONSE_EMPTY,
    LLMCallProps,
    LLMHubAPI,
    LLMHubMetadata,
    LLMPayload,
    LLMRequest,
    LLMResponseExt,
    ResourceId,
)
from mmar_utils import limit_concurrency, noop_decorator, pretty_line, retry_on_cond_and_ex
from requests.exceptions import ConnectTimeout

from mmar_llm.endpoints import find_llm_endpoint
from mmar_llm.gigachat_endpoint import GigaChatEndpoint
from mmar_llm.llm_endpoint import LLMEndpoint
from mmar_llm.llm_hub_config import LLMConfig, LLMHubConfig
from mmar_llm.models import ServiceUnavailableException
from mmar_llm.openrouter_endpoint import OpenRouterEndpoint

ENDPOINTS_CAPABILITIES: dict[str, type | tuple[type]] = {
    "image": OpenRouterEndpoint,
    "file": GigaChatEndpoint,
}
IMAGE_MIME_TYPES = {
    "jpg": "image/jpeg",
    "png": "image/png",
}
NA = "NOT AVAILABLE"
Limiter = Callable[..., Callable]


def is_ok_text_response(result: str) -> bool:
    return bool(result and result.strip())


def is_ok_embedding(embedding) -> bool:
    return any(map(abs, embedding))


def _parse_prompt_for_image(payload: LLMPayload) -> str:
    messages = payload.messages
    m_len = len(messages)
    if m_len == 0:
        return ""
    if m_len > 1:
        logger.warning(f"One message expected, but passed {m_len}")
    msg = messages[-1]
    return msg.content


class LLMHub(LLMHubAPI):
    def __init__(self, config: LLMHubConfig):
        config_llm = config.llm
        self.config_llm: LLMConfig = config_llm
        self._lock = Lock()
        self.limiters: dict[str, Limiter] = {}

        self.wait_seconds = config_llm.wait_seconds_on_llm_retry
        self.endpoint_keys = [ep.key for ep in config_llm.endpoints]
        self.validate_endpoints = getattr(config, "validate_endpoints", True)

        if config_llm.warmup:
            eks_loaded = {ek for ek in self.endpoint_keys if self._get_endpoint(ek) is not None}
            endpoints_pretty = ", ".join(eks_loaded)
            logger.info(f"Ready endpoints: {endpoints_pretty}")

            eks_not_loaded = {ek for ek in self.endpoint_keys if ek not in eks_loaded}
            if eks_not_loaded:
                na_endpoints_pretty = ", ".join(eks_not_loaded)
                logger.warning(f"Not available endpoints: {na_endpoints_pretty}")
        else:
            endpoints_pretty = ", ".join(self.endpoint_keys)
            logger.info(f"Endpoints: {endpoints_pretty}")

        self.default_ek = config_llm.default_endpoint_key
        self.default_image_ek = config_llm.default_image_endpoint_key
        self.default_file_ek = config_llm.default_file_endpoint_key
        files_dir = getattr(config, "files_dir", None)
        self.file_storage = FileStorage.create(files_dir)

    @cache
    def _get_endpoint(self, ek: str, capability: str | None = None) -> LLMEndpoint | None:
        try:
            ep_configs = [epc for epc in self.config_llm.endpoints if epc.key == ek]
            if len(ep_configs) == 0:
                logger.warning(f"Not found endpoint with key={ek}")
                return None
            if len(ep_configs) > 1:
                logger.warning(f"Found many endpoints configs: {ep_configs}")
            ep_config = ep_configs[0]
            ep_descriptor = ep_config.descriptor
            ep_class = find_llm_endpoint(ep_descriptor)
            if ep_class is None:
                logger.warning(f"Not found endpoint for descriptor={ep_descriptor}")
                return None
            ep = ep_class(**ep_config.args)
            # todo fix
            ep._key = ek
            if self.validate_endpoints:
                if not isinstance(ep, LLMEndpoint):
                    raise ValueError(f"Expected LLMEndpoint, but found {type(ep)}: {ep}")
                # todo validate methods
            if ep is None:
                logger.warning(f"Not found endpoint with key={ek}")
                return None
            if not capability:
                return ep
            cls = ENDPOINTS_CAPABILITIES[capability]
            if isinstance(ep, cls):
                return ep
            logger.warning(f"Expected endpoint {cls}, but found: {type(ep)}")
            return None
        except (ServiceUnavailableException, ConnectTimeout) as ex:
            logger.error(f"Failed to create endpoint with key={ek}: {ex}")
            return None

    def _get_endpoint_or_default(self, ek: str, default_ek: str, *, capability: str = "") -> LLMEndpoint | None:
        return (ek and self._get_endpoint(ek, capability)) or self._get_endpoint(default_ek, capability)

    def _get_response_from_image(self, payload: LLMPayload, resource_id, props: LLMCallProps) -> str:
        ek = props.endpoint_key

        dtype = self.file_storage.get_dtype(resource_id)
        mimetype: str | None = mimetypes.guess_type(resource_id)[0] or (dtype and IMAGE_MIME_TYPES[dtype])
        if not mimetype:
            logger.error(f"Failed to derive mimetype: {mimetype}")
            return ""

        sent: str = _parse_prompt_for_image(payload)
        ep = self._get_endpoint_or_default(ek, self.default_image_ek, capability="image")
        if ep is None:
            logger.error(f"Failed to get image endpoint for keys=('{ek}', '{self.default_image_ek}')")
            return ""

        file: bytes = self.file_storage.download(resource_id)
        get_image_response = getattr(ep, "get_image_response", None)
        if not get_image_response:
            logger.error(f"Not image endpoint: {ek}: {type(ep)}")
            return ""
        response_image: str = get_image_response(bytesimage=file, sentences=sent, mimetype=mimetype)
        logger.debug(f"Image response from {ep.__repr__()}: `{pretty_line(response_image)}`")
        return response_image

    def _upload_file(self, endpoint: LLMEndpoint, resource_id: ResourceId) -> str | None:
        # todo don't upload already loaded file
        # todo try-catch?
        resource_path = Path(resource_id)
        if not resource_path.exists():
            logger.error(f"Can not found resource_id={resource_id}")
            return None
        with resource_path.open("rb") as file_handle:
            upload_file = getattr(endpoint, "upload_file")
            if not upload_file:
                logger.warning(f"Not file endpoint: {endpoint}")
                return None
            uploaded_file = upload_file(file=file_handle)
            file_id = uploaded_file.id_
            logger.info(f"Uploaded file: {resource_id} -> {file_id}")
            return file_id

    def _get_limit(self, endpoint_key: str) -> int:
        ec = self.config_llm.get_endpoint_config(endpoint_key)
        if not ec:
            return self.config_llm.default_concurrency_limit
        return ec.concurrency_limit

    def _get_limiter(self, endpoint_key: str) -> Limiter:
        with self._lock:
            limiter = self.limiters.get(endpoint_key)
            if limiter:
                return limiter

            limit = self._get_limit(endpoint_key)
            if limit == -1:
                logger.info(f"Creating dummy limiter for endpoint_key={endpoint_key}")
                limiter = noop_decorator
            else:
                logger.info(f"Creating limiter({limit}) for endpoint_key={endpoint_key}")
                limiter = limit_concurrency(limit)
            self.limiters[endpoint_key] = limiter
            return limiter

    def _get_response_from_payload(
        self, endpoint: LLMEndpoint, payload: LLMPayload, props: LLMCallProps = LCP
    ) -> LLMResponseExt:
        ek = props.endpoint_key
        ek_pretty = f"(endpoint_key={props.endpoint_key})" if ek else ""
        # todo fix: it's flaky that
        retrier = retry_on_cond_and_ex(
            title=f"#get_response{ek_pretty}",
            attempts=props.attempts,
            wait_seconds=self.wait_seconds,
            logger=logger,
            condition=is_ok_text_response,
        )
        get_response = endpoint.get_response
        # endpoint._key is always set by _get_endpoint, but use str conversion for type safety
        limiter = self._get_limiter(str(endpoint._key))  # type: ignore[arg-type]
        get_response = limiter(get_response)
        get_response = retrier(get_response)

        payload_dict = {"messages": payload.model_dump()["messages"]}
        if payload.attachments:
            payload_dict["attachments"] = payload.attachments
        text = get_response(request=payload_dict) or ""
        return LLMResponseExt(text=text)

    def _get_response_ext(self, payload: LLMPayload, props: LLMCallProps) -> LLMResponseExt:
        ek = props.endpoint_key
        resource_id: str | None = payload.get_resource_id()

        if not resource_id:
            endpoint_b: LLMEndpoint | None = self._get_endpoint_or_default(ek, self.default_ek)
            if endpoint_b is None:
                logger.error(f"Failed to find endpoint: {ek}, default: {self.default_ek}")
                return RESPONSE_EMPTY
            return self._get_response_from_payload(endpoint_b, payload, props)

        dtype = self.file_storage.get_dtype(resource_id)

        if dtype in {"jpg", "png"}:
            text = self._get_response_from_image(payload, resource_id, props)
            return LLMResponseExt(text=text)

        if dtype in {"pdf", "csv", "txt"}:
            endpoint_f = self._get_endpoint_or_default(ek, self.default_file_ek, capability="file")
            if endpoint_f is None:
                logger.error(f"Failed to get file endpoint for keys=({ek}, {self.default_file_ek}")
                return RESPONSE_EMPTY

            # todo don't upload already loaded file
            # todo try-catch?
            file_id = self._upload_file(endpoint_f, resource_id)
            if file_id:
                payload_with_file = payload.with_attachments([[file_id]])
                logger.info(f"Sending request with file: {payload_with_file}")
                return self._get_response_from_payload(endpoint_f, payload_with_file, props)
            else:
                return self._get_response_from_payload(endpoint_f, payload, props)

        endpoint: LLMEndpoint | None = self._get_endpoint_or_default(ek, self.default_ek)
        if endpoint is None:
            logger.error(f"Failed to get file endpoint for keys=({ek}, {self.default_file_ek}")
            return RESPONSE_EMPTY

        return self._get_response_from_payload(endpoint, payload, props)

    # API

    def get_metadata(self) -> LLMHubMetadata:
        return self.config_llm.as_metadata()

    def get_response(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> str:
        response_ext = self.get_response_ext(request=request, props=props)
        return response_ext.text

    def get_response_ext(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> LLMResponseExt:
        start = time.time()
        payload: LLMPayload = LLMPayload.parse(request)
        response = self._get_response_ext(payload, props)
        elapsed = time.time() - start
        ek_pretty = f", endpoint_key={props.endpoint_key}" if props.endpoint_key else ""
        # todo fix: also show real time, without retries and waits
        logger.info(f"Ready in {elapsed:.2f} seconds ( {payload.show_pretty(detailed=True)}{ek_pretty} )")
        return response

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        ek = props.endpoint_key
        endpoint: LLMEndpoint | None = self._get_endpoint_or_default(ek, self.default_ek)
        if endpoint is None:
            logger.error(f"Failed to get endpoint for keys=({ek}, {self.default_file_ek}")
            return None
        # todo move to library
        prompt_pretty = pretty_line(prompt)
        retrier = retry_on_cond_and_ex(
            title=f"#get_embedding(endpoint_key={props.endpoint_key}), prompt={prompt_pretty}",
            attempts=props.attempts,
            condition=is_ok_embedding,
            wait_seconds=self.wait_seconds,
            logger=logger,
        )
        get_embedding = retrier(endpoint.get_embedding)
        return get_embedding(prompt=prompt)
