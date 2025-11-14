import mimetypes
import time
from functools import cache
from pathlib import Path

from loguru import logger
from mmar_llm import AbstractEntryPoint, EntrypointsAccessor, GigaChatEntryPoint, OpenRouterEntryPoint
from mmar_llm.models import ServiceUnavailableException
from mmar_mapi import FileStorage
from mmar_mapi.api import (
    LCP,
    RESPONSE_EMPTY,
    EntrypointInfo,
    EntrypointsConfig,
    LLMAccessorAPI,
    LLMCallProps,
    Message,
    Payload,
    Request,
    ResourceId,
    ResponseExt,
)
from mmar_utils import pretty_line, retry_on_cond
from requests.exceptions import ConnectTimeout

from src.config import Config

ENTRYPOINTS_CAPABILITIES: dict[str, type | tuple[type]] = {
    "image": OpenRouterEntryPoint,
    "file": GigaChatEntryPoint,
}
IMAGE_MIME_TYPES = {
    "jpg": "image/jpeg",
    "png": "image/png",
}
NA = "NOT AVAILABLE"


def _parse_messages(request: Request) -> list[Message]:
    if isinstance(request, str):
        return [Message(role="user", content=request)]
    elif isinstance(request, list) and all(isinstance(msg, Message) for msg in request):
        return request
    elif isinstance(request, Payload):
        return request.messages
    else:
        raise ValueError(f"Bad request type {type(request)}: {request}")


def _parse_resource_id(request: Request) -> str | None:
    if isinstance(request, str):
        return None
    elif isinstance(request, list) and all(isinstance(msg, Message) for msg in request):
        return None
    elif isinstance(request, Payload):
        attachments = request.attachments
        if not attachments:
            return None
        # todo log message that many attachments passed
        resource_id = attachments[0][0]
        return resource_id
    else:
        raise ValueError(f"Bad request type {type(request)}: {request}")


def _parse_prompt_for_image(payload: Payload) -> str:
    messages = payload.messages
    m_len = len(messages)
    if m_len == 0:
        return ""
    if m_len > 1:
        logger.warning(f"One message expected, but passed {m_len}")
    msg = messages[-1]
    return msg.content


class LLMAccessor(LLMAccessorAPI):
    def __init__(self, config: Config):
        self.config = config
        self.entrypoint_keys = list(config.llm.entrypoints.keys())

        self.entrypoint_accessor = EntrypointsAccessor(config.llm)

        if config.warmup_entrypoints:
            eks_loaded = {ek for ek in self.entrypoint_keys if self._get_entrypoint(ek) is not None}
            entrypoints_pretty = ", ".join(eks_loaded)
            logger.info(f"Ready entrypoints: {entrypoints_pretty}")
            na_entrypoints_pretty = ", ".join(ek for ek in self.entrypoint_keys if ek not in eks_loaded)
            logger.warning(f"Not available entrypoints: {na_entrypoints_pretty}")
        else:
            entrypoints_pretty = ", ".join(self.entrypoint_keys)
            logger.info(f"Entrypoints: {entrypoints_pretty}")

        self.default_ek = config.llm.default_entrypoint_key
        # todo fix
        self.default_image_ek = config.llm.default_image_entrypoint_key
        self.default_file_ek = config.llm.default_file_entrypoint_key
        self.file_storage = FileStorage(config.files_dir)

    @cache
    def _get_entrypoint(self, ek: str, capability: str | None = None) -> AbstractEntryPoint | None:
        try:
            ep = self.entrypoint_accessor.get(ek)
            if ep is None:
                logger.warning(f"Not found entrypoint with key={ek}")
                return None
            if not capability:
                return ep
            cls = ENTRYPOINTS_CAPABILITIES[capability]
            if isinstance(ep, cls):
                return ep
            logger.warning(f"Expected entrypoint {cls}, but found: {type(ep)}")
            return None
        except (ServiceUnavailableException, ConnectTimeout) as ex:
            logger.error(f"Failed to create entrypoint with key={ek}: {ex}")
            return None

    def _get_entrypoint_or_default(
        self, ek: str, default_ek: str, *, capability: str = ""
    ) -> AbstractEntryPoint | None:
        return (ek and self._get_entrypoint(ek, capability)) or self._get_entrypoint(default_ek, capability)

    def _get_response_from_image(self, payload: Payload, resource_id, props: LLMCallProps) -> str:
        ek = props.entrypoint_key

        dtype = self.file_storage.get_dtype(resource_id)
        mimetype: str | None = mimetypes.guess_type(resource_id)[0] or (dtype and IMAGE_MIME_TYPES[dtype])
        if not mimetype:
            logger.error(f"Failed to derive mimetype: {mimetype}")
            return ""

        sent: str = _parse_prompt_for_image(payload)
        ep = self._get_entrypoint_or_default(ek, self.default_image_ek, capability="image")
        if ep is None:
            logger.error(f"Failed to get image entrypoint for keys=({ek}, {self.default_image_ek}")
            return ""

        file: bytes = self.file_storage.download(resource_id)
        get_image_response = getattr(ep, "get_image_response", None)
        if not get_image_response:
            logger.error(f"Not image entrypoint: {ek}: {type(ep)}")
            return ""
        response_image: str = get_image_response(bytesimage=file, sentences=sent, mimetype=mimetype)
        logger.debug(f"Image response from {ep.__repr__()}: `{pretty_line(response_image)}`")
        return response_image

    def _upload_file(self, entrypoint: AbstractEntryPoint, resource_id: ResourceId) -> str | None:
        # todo don't upload already loaded file
        # todo try-catch?
        resource_path = Path(resource_id)
        if not resource_path.exists():
            logger.error(f"Can not found resource_id={resource_id}")
            return None
        with resource_path.open("rb") as file_handle:
            upload_file = getattr(entrypoint, "upload_file")
            if not upload_file:
                logger.warning(f"Not file entrypoint: {entrypoint}")
                return None
            uploaded_file = upload_file(file=file_handle)
            file_id = uploaded_file.id_
            logger.info(f"Uploaded file: {resource_id} -> {file_id}")
            return file_id

    def _get_response_from_payload(
        self, entrypoint: AbstractEntryPoint, payload: Payload, props: LLMCallProps = LCP
    ) -> ResponseExt:
        retrier = retry_on_cond(title=f"#get_response(entrypoint_key={props.entrypoint_key})", attempts=props.attempts)
        get_response_by_payload = retrier(entrypoint.get_response_by_payload)
        # todo fix
        payload = payload.model_dump()['messages']
        logger.debug(f"Payload: {payload}")
        text = get_response_by_payload(payload) or ""
        return ResponseExt(text=text)

    # API

    def get_entrypoints_config(self) -> EntrypointsConfig:
        e_configs = self.config.llm.entrypoints.values()
        entrypoints = [EntrypointInfo(entrypoint_key=epc.key, caption=epc.caption) for epc in e_configs]
        return EntrypointsConfig(default_entrypoint_key=self.default_ek, entrypoints=entrypoints)

    def get_entrypoint_keys(self) -> list[str]:
        return self.entrypoint_keys

    def get_response(self, *, request: Request, props: LLMCallProps = LCP) -> str:
        response_ext = self.get_response_ext(request=request, props=props)
        return response_ext.text

    def get_response_ext(self, *, request: Request, props: LLMCallProps = LCP) -> ResponseExt:
        start = time.time()
        response = self._get_response_ext(request, props)
        elapsed = time.time() - start
        logger.info(f"Ready in {elapsed:.2f} seconds")
        return response

    def _get_response_ext(self, request: Request, props: LLMCallProps) -> ResponseExt:
        ek = props.entrypoint_key
        messages: list[Message] = _parse_messages(request)
        payload = Payload(messages=messages)
        resource_id: str | None = _parse_resource_id(request)

        if not resource_id:
            entrypoint_b: AbstractEntryPoint | None = self._get_entrypoint_or_default(ek, self.default_ek)
            if entrypoint_b is None:
                logger.error(f"Failed to find entrypoint: {ek}, default: {self.default_ek}")
                return RESPONSE_EMPTY
            return self._get_response_from_payload(entrypoint_b, payload, props)

        dtype = self.file_storage.get_dtype(resource_id)

        if dtype in {"jpg", "png"}:
            text = self._get_response_from_image(payload, resource_id, props)
            return ResponseExt(text=text)

        if dtype in {"pdf", "csv", "txt"}:
            entrypoint_f = self._get_entrypoint_or_default(ek, self.default_file_ek, capability="file")
            if entrypoint_f is None:
                logger.error(f"Failed to get file entrypoint for keys=({ek}, {self.default_file_ek}")
                return RESPONSE_EMPTY

            # todo don't upload already loaded file
            # todo try-catch?
            file_id = self._upload_file(entrypoint_f, resource_id)
            if file_id:
                payload_with_file = payload.with_attachments([[file_id]])
                logger.info(f"Sending request with file: {payload_with_file}")
                return self._get_response_from_payload(entrypoint_f, payload_with_file, props)
            else:
                return self._get_response_from_payload(entrypoint_f, payload, props)

        entrypoint: AbstractEntryPoint | None = self._get_entrypoint_or_default(ek, self.default_ek)
        if entrypoint is None:
            logger.error(f"Failed to get file entrypoint for keys=({ek}, {self.default_file_ek}")
            return RESPONSE_EMPTY

        return self._get_response_from_payload(entrypoint, payload, props)

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        ek = props.entrypoint_key
        entrypoint: AbstractEntryPoint | None = self._get_entrypoint_or_default(ek, self.default_ek)
        if entrypoint is None:
            logger.error(f"Failed to get entrypoint for keys=({ek}, {self.default_file_ek}")
            return None
        # todo move to library
        prompt_pretty = pretty_line(prompt)
        retrier = retry_on_cond(
            title=f"#get_embedding(entrypoint_key={props.entrypoint_key}), prompt={prompt_pretty}",
            attempts=props.attempts,
            condition=lambda embedding: any(map(abs, embedding)),
        )
        get_embedding = retrier(entrypoint.get_embedding)
        return get_embedding(prompt)
