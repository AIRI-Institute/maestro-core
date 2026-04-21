from http import HTTPStatus
from typing import Annotated

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from mmar_mapi import Context, FileStorage

from gateway.config import Config
from gateway.fastapi_errors import ERR_STATUSES, FailedToUploadException, FileNotFoundException, MalformedException
from gateway.fastapi_files import load_file, make_streaming_response
from gateway.legacy import as_file_data, upload_resource_maybe
from gateway.maestro_gateway import MaestroGateway
from gateway.models import (
    ChatRequestMessages,
    ChatResponse,
    CreateResponse,
    DBChatPreviews,
    DomainsResponse,
    FileData,
    HistoryResponse,
    ModelsResponse,
    TracksResponse,
    UploadManyRequest,
    UploadResponse,
)
from gateway.validators import validate_client, validate_no_underscores

D = Depends
ClientIdHeader = Annotated[Annotated[str, D(validate_client)], Header()]
router = APIRouter(responses=ERR_STATUSES)


@router.get("/api/v2/chats")
@inject
async def get_chat_previews(
    client_id: ClientIdHeader,
    user_id: str,
    gateway: FromDishka[MaestroGateway],
) -> DBChatPreviews:
    return await gateway.get_chat_previews(client_id, user_id)


@router.post("/api/v3/chats")
@inject
async def create(
    client_id: ClientIdHeader,
    context: Context,
    gateway: FromDishka[MaestroGateway],
) -> CreateResponse:
    if context.client_id == "":
        context.client_id = client_id
    elif client_id != context.client_id:
        err = f"client-id from context ({context.client_id}) and header ({client_id}) are not aligned"
        raise MalformedException(err)

    validate_no_underscores(context, "client_id")
    validate_no_underscores(context, "user_id")
    validate_no_underscores(context, "session_id")

    response: CreateResponse = await gateway.create_chat(context=context)
    return response


@router.get("/api/v3/chats/{chat_id}")
@inject
async def get_chat(
    client_id: ClientIdHeader,
    chat_id: str,
    gateway: FromDishka[MaestroGateway],
) -> HistoryResponse:
    err, chat = await gateway.get_chat(chat_id=chat_id)
    if err:
        logger.error(f"Failed to delete chat_id={chat_id}: {err}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND)
    assert chat is not None
    messages = [msg for msg in chat.messages if msg.is_human or msg.is_ai]
    return HistoryResponse(chat_id=chat_id, messages=messages)


@router.post("/api/v3/chats/{chat_id}")
@inject
async def send(
    client_id: ClientIdHeader,
    chat_id: str,
    request: ChatRequestMessages,
    gateway: FromDishka[MaestroGateway],
) -> ChatResponse:
    messages = request.messages
    if len(messages) != 1:
        raise MalformedException(detail=f"One message expected, found: {len(messages)}")
    msg = messages[0]

    response: ChatResponse = await gateway.send_message_by_chat_id(chat_id=chat_id, msg=msg)
    return response


@router.delete("/api/v3/chats/{chat_id}")
@inject
async def delete_chat(
    client_id: ClientIdHeader,
    chat_id: str,
    gateway: FromDishka[MaestroGateway],
) -> None:
    err = await gateway.delete_chat(chat_id)
    if err:
        logger.error(f"Failed to delete chat_id={chat_id}: {err}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND)


@router.post("/api/v2/files/upload")
@inject
async def upload(
    client_id: ClientIdHeader,
    file: UploadFile | None = None,
    config: FromDishka[Config] = Depends(),
    file_storage: FromDishka[FileStorage] = Depends(),
) -> UploadResponse:
    f_data: FileData | None = await load_file(config, file) if file else None
    if not f_data:
        raise FileNotFoundException()
    f_name, f_content = f_data
    named_resource_id = upload_resource_maybe(file_storage, f_data)
    if not named_resource_id:
        raise FailedToUploadException()
    resource_name, resource_id = named_resource_id
    logger.info(f"Uploaded resource to {resource_id}, size: {len(f_content)}, name: {resource_name}")
    return UploadResponse(resource_id=resource_id, resource_name=resource_name)


@router.post("/api/v1/files/upload_dir")
@inject
async def upload_dir(
    client_id: ClientIdHeader,
    request: UploadManyRequest,
    file_storage: FromDishka[FileStorage] = Depends(),
) -> UploadResponse:
    logger.info(f"Files received: {request.resource_ids}")
    rid = file_storage.upload_dir(list(request.resource_ids))
    return UploadResponse(resource_id=rid, error=None)


@router.get("/api/v2/files/download_text")
@inject
async def download_text(
    client_id: ClientIdHeader,
    resource_id: str,
    file_storage: FromDishka[FileStorage] = Depends(),
) -> str:
    res: str = file_storage.download_text(resource_id)
    return res


@router.get("/api/v2/files/download_bytes")
@inject
async def download_bytes(
    client_id: ClientIdHeader,
    resource_id: str,
    file_storage: FromDishka[FileStorage] = Depends(),
) -> StreamingResponse:
    f_data = as_file_data(file_storage, resource_id)
    res = make_streaming_response(f_data)
    return res


@router.get("/api/v3/info/domains")
@inject
async def get_domains(
    client_id: ClientIdHeader = "",
    language_code: str = "ru",
    gateway: FromDishka[MaestroGateway] = Depends(),
) -> DomainsResponse:
    domains = await gateway.get_domains(language_code=language_code, client_id=client_id)
    return DomainsResponse(domains=domains)


@router.get("/api/v3/info/tracks")
@inject
async def get_tracks(
    client_id: ClientIdHeader = "",
    language_code: str = "ru",
    gateway: FromDishka[MaestroGateway] = Depends(),
) -> TracksResponse:
    tracks = await gateway.get_tracks(language_code=language_code, client_id=client_id)
    return TracksResponse(tracks=tracks)


@router.get("/api/v3/models")
@inject
async def get_models(
    gateway: FromDishka[MaestroGateway] = Depends(),
) -> ModelsResponse:
    return await gateway.get_models()


@router.get("/api/health/readiness", description="Is app ready")
async def readiness() -> bool:
    return True


@router.get("/api/health/liveness", description="Is app live")
async def liveness() -> bool:
    return True
