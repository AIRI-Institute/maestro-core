import asyncio
import time
from functools import cache, partial, wraps
from types import SimpleNamespace

from aiohttp import ClientConnectionError, ClientResponseError, FormData
from loguru import logger
from mmar_mapi import AIMessage, Context, DomainInfo, FileStorage, HumanMessage, ResourceId, TrackInfo, make_content
from mmar_utils import on_error_log_and_none, remove_prefix_if_present, retry_on_ex

from mmar_mcli.io_aiohttp import make_file_form_data, request_with_session
from mmar_mcli.models import FileData, MaestroClientConfig, MessageData, ModelsResponse, RequestCall

ROUTES = SimpleNamespace(
    send="api/v0/send",
    download="api/v2/files/download_bytes",
    upload="api/v2/files/upload",
    domains="api/v3/info/domains",
    tracks="api/v3/info/tracks",
    models="api/v3/models",
)
MESSAGE_START: MessageData = make_content(text="/start"), None
POST_ERRORS = (asyncio.TimeoutError, ClientConnectionError, ClientResponseError)


def fix_maestro_address(maestro_address: str) -> str:
    if not maestro_address.startswith("https:"):
        if maestro_address.startswith(":"):
            maestro_address = f"localhost{maestro_address}"
        maestro_address = "http://" + remove_prefix_if_present(maestro_address, "http://")
    return maestro_address


class RemoteStorager:
    def __init__(self, request: RequestCall, url_base: str, client_id: str):
        self.request = request
        self.url_download = url_base.replace(ROUTES.send, ROUTES.download)
        self.url_upload = url_base.replace(ROUTES.send, ROUTES.upload)
        self.client_id = client_id
        self.headers = {"client-id": client_id}

    async def upload_async(self, content: bytes | str, fname: str) -> ResourceId:
        content = content if isinstance(content, bytes) else content.encode()
        data: FormData = make_file_form_data((fname, content))
        response_data = await self.request(method="post", url=self.url_upload, headers=self.headers, data=data)
        if not isinstance(response_data, dict):
            raise ValueError(f"POST {self.url_upload}: expected json, found {type(response_data)}")
        resource_id: str = response_data.get("ResourceId") or response_data.get("resource_id")
        return resource_id

    async def download_async(self, resource_id: str) -> bytes:
        params = dict(resource_id=resource_id)
        resource_bytes = await self.request(method="get", url=self.url_download, params=params, headers=self.headers)
        if not isinstance(resource_bytes, bytes):
            raise ValueError(f"GET {self.url_download}: expected bytes, found {type(resource_bytes)}")
        return resource_bytes


class MaestroClientI:
    async def send_simple(self, context: Context, msg_data: MessageData | str) -> list[MessageData]:
        pass

    async def upload_resource(self, file_data: FileData, client_id: str) -> str | None:
        pass

    async def download_resource(self, resource_id: str, client_id: str) -> bytes:
        pass

    async def get_domains(self, language_code: str, client_id: str) -> list[DomainInfo]:
        pass

    async def get_tracks(self, language_code: str, client_id: str) -> list[TrackInfo]:
        pass

    async def get_models(self) -> ModelsResponse:
        pass

    async def send(self, context: Context, msg: HumanMessage) -> list[AIMessage] | None:
        pass


class MaestroClient(MaestroClientI):
    def __init__(self, config: MaestroClientConfig | SimpleNamespace):
        """Initialize MaestroClient with a configuration.

        Args:
            config: Configuration object. Can be:
                - MaestroClientConfig: Proper configuration dataclass
                - SimpleNamespace: For backward compatibility
                - Any object with matching attributes
        """
        cfg = MaestroClientConfig.create(config)

        self.url = fix_maestro_address(cfg.addresses__maestro) + "/" + ROUTES.send
        logger.info(f"Creating client, maestro URL: {self.url}")

        request = partial(request_with_session, timeout=cfg.timeout, headers_extra=cfg.headers_extra)
        request = wraps(request_with_session)(request)
        if cfg.with_retries:
            request = retry_on_ex(attempts=3, wait_seconds=1, catch=POST_ERRORS, logger=logger)(request)
        request = on_error_log_and_none(logger.exception)(request)

        self.request = request
        self.msg_data_response_error: MessageData = make_content(text=cfg.res.error), None

        self.file_storage: FileStorage | None = cfg.files_dir and FileStorage(cfg.files_dir)

    @cache
    def get_file_storage(self, client_id: str) -> FileStorage:
        return self.file_storage or RemoteStorager(self.request, self.url, client_id)

    async def send_simple(self, context: Context, msg_data: MessageData | str) -> list[MessageData]:
        start = time.time()
        msg_datas_response = await self._send_simple(context, msg_data)
        elapsed = time.time() - start

        entrypoint_key = (context.extra or {}).get("entrypoint_key", "")
        cd = f"{context.track_id}.{context.session_id}.{entrypoint_key}"
        log_suffix = " (NULL!)" if not msg_datas_response else ""
        logger.info(f"BotResponse processing time for {cd}: {elapsed:.2f} s{log_suffix}")

        msg_datas_response = msg_datas_response or [self.msg_data_response_error]
        return msg_datas_response

    async def upload_resource(self, file_data: FileData, client_id: str) -> str | None:
        file_name, file_bytes = file_data
        resourse_id: str = await self.get_file_storage(client_id).upload_async(file_bytes, file_name)
        logger.debug(f"Uploaded resource with name '{file_name}' to '{resourse_id}'")
        return resourse_id

    async def download_resource(self, resource_id: str, client_id: str) -> bytes:
        res: bytes = await self.get_file_storage(client_id).download_async(resource_id)
        return res

    async def get_domains(self, language_code: str, client_id: str) -> list[DomainInfo]:
        url = self.url.replace(ROUTES.send, ROUTES.domains)
        params = {"language_code": language_code, "client_id": client_id}
        response_data = await self.request(method="get", url=url, params=params)
        if not isinstance(response_data, dict):
            raise ValueError(f"GET {url}: expected dict, found {type(response_data)}")
        domains_raw = response_data.get("domains", response_data)
        return [DomainInfo.model_validate(d) for d in domains_raw]

    async def get_tracks(self, language_code: str, client_id: str) -> list[TrackInfo]:
        url = self.url.replace(ROUTES.send, ROUTES.tracks)
        params = {"language_code": language_code, "client_id": client_id}
        response_data = await self.request(method="get", url=url, params=params)
        if not isinstance(response_data, dict):
            raise ValueError(f"GET {url}: expected dict, found {type(response_data)}")
        tracks_raw = response_data.get("tracks", response_data)
        return [TrackInfo.model_validate(t) for t in tracks_raw]

    async def get_models(self) -> ModelsResponse:
        url = self.url.replace(ROUTES.send, ROUTES.models)
        response_data = await self.request(method="get", url=url)
        if not isinstance(response_data, dict):
            raise ValueError(f"GET {url}: expected dict, found {type(response_data)}")
        return ModelsResponse.model_validate(response_data)

    async def _download_file_data_maybe(self, msg: AIMessage, client_id: str) -> FileData | None:
        resource_id = msg.resource_id
        if not resource_id:
            return None
        logger.info(f"Downloading resource: {resource_id}")
        resource_name = msg.resource_name
        if not resource_name:
            resource_ext = resource_id.split(".")[-1]
            resource_name = f"result.{resource_ext}"
        resource_bytes = await self.download_resource(resource_id, client_id)
        return resource_name, resource_bytes

    async def _send_simple(self, context: Context, msg_data: MessageData | str) -> list[MessageData] | None:
        if isinstance(msg_data, str):
            msg_data = msg_data, None
        content, file_data = msg_data

        resource_id = file_data and await self.upload_resource(file_data, context.client_id)
        content = make_content(content=content, resource_id=resource_id)
        msg = HumanMessage(content=content)
        ai_messages = await self.send(context, msg)
        if not ai_messages:
            return None
        download = partial(self._download_file_data_maybe, client_id=context.client_id)
        res = [(ai_msg.content, await download(ai_msg)) for ai_msg in ai_messages]
        return res

    async def send(self, context: Context, msg: HumanMessage) -> list[AIMessage] | None:
        dict_user_message = msg.model_dump()
        dict_ctx = context.model_dump()
        data_json = {"context": dict_ctx, "messages": [dict_user_message]}
        headers = {"client-id": context.client_id}
        try:
            response_data = await self.request(method="post", url=self.url, json=data_json, headers=headers)
        except Exception:
            logger.exception(f"Failed to send request {msg}")
            return None
        if response_data is None:
            return None
        logger.trace(f"Response data: {response_data}")
        response_messages_raw = response_data["response_messages"]
        ai_messages = list(map(AIMessage.model_validate, response_messages_raw))
        return ai_messages
