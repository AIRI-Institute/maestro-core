from typing import Annotated

from fastapi import APIRouter, Depends, Header

from gateway.dependencies import Deps
from gateway.fastapi_errors import ERR_STATUSES
from gateway.legacy import ChatRequestOld, ChatResponseOld
from gateway.models import ChatResponse
from gateway.validators import validate_client

router_legacy = APIRouter(responses=ERR_STATUSES, tags=["deprecated"])
D = Depends
ClientIdHeader = Annotated[Annotated[str, D(validate_client)], Header()]


def validate_client_request_old(request: ChatRequestOld, deps: Deps = D()) -> ChatRequestOld:
    validate_client(request.context.client_id)
    return request


ValidatedChatRequestOld = Annotated[ChatRequestOld, D(validate_client_request_old)]


@router_legacy.post("/api/v0/send")
async def get_response_v6(request: ValidatedChatRequestOld, deps: Deps = D()) -> ChatResponseOld:
    response: ChatResponse = await deps.gateway.send_message_by_chat_request_old(request)
    res = ChatResponseOld(
        context=request.context,
        messages=request.messages,
        response_messages=response.response_messages,
    )
    return res
