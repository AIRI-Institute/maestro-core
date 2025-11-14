from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from mmar_mapi import Context

from src.dependencies import Deps
from src.fastapi_errors import ERR_STATUSES, FailedToUploadException, FileNotFoundException, MalformedException
from src.fastapi_files import load_file, make_streaming_response
from src.legacy import as_file_data, upload_resource_maybe
from src.models import (
    ChatRequestMessages,
    ChatResponse,
    CreateResponse,
    DBChatPreviews,
    DomainsResponse,
    EntrypointsResponse,
    FileData,
    HistoryResponse,
    TracksResponse,
    UploadManyRequest,
    UploadResponse,
)
from src.utils_funcs import partial_hide
from src.validators import validate_client

D = Depends
ClientIdHeader = Annotated[Annotated[str, D(validate_client)], Header()]
router = APIRouter(responses=ERR_STATUSES)


@router.get("/api/v2/chats")
async def get_chat_previews(client_id: ClientIdHeader, user_id: str, deps: Deps = D()) -> DBChatPreviews:
    return await deps.gateway.get_chat_previews(client_id, user_id)


@router.post("/api/v3/chats")
async def create(client_id: ClientIdHeader, context: Context, deps: Deps = D()) -> CreateResponse:
    # todo fix: is it needed to specify client_id here?
    if context.client_id == "":
        context.client_id = client_id
    elif client_id != context.client_id:
        err = f"client-id from context ({context.client_id}) and header ({client_id}) are not aligned"
        raise MalformedException(err)
    # todo fix: ensure that session_id is okeish
    # todo fix: validate that not `_` inside
    context_fix = Context(**context.model_dump())

    response: CreateResponse = await deps.gateway.create_chat(context=context_fix)
    return response


@router.get("/api/v3/chats/{chat_id}")
async def get_chat(client_id: ClientIdHeader, chat_id: str, deps: Deps = D()) -> HistoryResponse:
    err, chat = await deps.gateway.get_chat(chat_id=chat_id)
    if err:
        # todo fix: why NOT_FOUND? maybe other error?
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=err)
    messages = [msg for msg in chat.messages if msg.is_human or msg.is_ai]
    return HistoryResponse(chat_id=chat_id, messages=messages)


@router.post("/api/v3/chats/{chat_id}")
async def send(client_id: ClientIdHeader, chat_id: str, request: ChatRequestMessages, deps: Deps = D()) -> ChatResponse:
    messages = request.messages
    if len(messages) != 1:
        raise MalformedException(detail=f"One message expected, found: {len(messages)}")
    msg = messages[0]

    response: ChatResponse = await deps.gateway.send_message_by_chat_id(chat_id=chat_id, msg=msg)
    return response


@router.delete("/api/v3/chats/{chat_id}")
async def delete_chat(client_id: ClientIdHeader, chat_id: str, deps: Deps = D()) -> None:
    err = await deps.gateway.delete_chat(chat_id)
    if err:
        # todo fix: why NOT_FOUND? maybe other error?
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=err)
    return None


@router.post("/api/v2/files/upload")
async def upload(client_id: ClientIdHeader, file: UploadFile | None = None, deps: Deps = D()) -> UploadResponse:
    f_data: FileData | None = await load_file(deps.config, file)
    if not f_data:
        raise FileNotFoundException()
    f_name, f_content = f_data
    named_resource_id = upload_resource_maybe(deps.file_storage, f_data)
    if not named_resource_id:
        raise FailedToUploadException()
    resource_name, resource_id = named_resource_id
    logger.info(f"Uploaded resource to {resource_id}, size: {len(f_content)}, name: {resource_name}")
    return UploadResponse(resource_id=resource_id, resource_name=resource_name)


@router.post("/api/v1/files/upload_dir")
async def upload_dir(client_id: ClientIdHeader, request: UploadManyRequest, deps: Deps = D()) -> UploadResponse:
    logger.info(f"Files received: {request.resource_ids}")
    rid = deps.file_storage.upload_dir(request.resource_ids)
    return UploadResponse(resource_id=rid, error=None)


@router.get("/api/v2/files/download_text")
async def download_text(client_id: ClientIdHeader, resource_id: str, deps: Deps = D()) -> str:
    res: str = deps.file_storage.download_text(resource_id)
    return res


@router.get("/api/v2/files/download_bytes")
async def download_bytes(client_id: ClientIdHeader, resource_id: str, deps: Deps = D()) -> StreamingResponse:
    f_data = as_file_data(deps.file_storage, resource_id)
    res = make_streaming_response(f_data)
    return res


@router.get("/api/v3/info/domains")
async def get_domains(client_id: ClientIdHeader = "", language_code: str = "ru", deps: Deps = D()) -> DomainsResponse:
    return await deps.gateway.get_domains(language_code=language_code, client_id=client_id)


@router.get("/api/v3/info/tracks")
async def get_tracks(client_id: ClientIdHeader = "", language_code: str = "ru", deps: Deps = D()) -> TracksResponse:
    return await deps.gateway.get_tracks(language_code=language_code, client_id=client_id)


@router.get("/api/v3/info/entrypoints")
async def get_entrypoints(cid: ClientIdHeader = "", language_code: str = "ru", d: Deps = D()) -> EntrypointsResponse:
    return await d.gateway.get_entrypoints(cid)


@router.get("/api/health/readiness", description="Is app ready")
async def readiness() -> bool:
    return True


@router.get("/api/health/liveness", description="Is app live")
async def liveness() -> bool:
    return True
