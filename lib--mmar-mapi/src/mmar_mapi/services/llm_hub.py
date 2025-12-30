from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from mmar_mapi import ChatMessage, BaseMessage


class LLMCallProps(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    endpoint_key: str = ""
    attempts: int = 1

    def with_endpoint_key(self, endpoint_key):
        return self.model_copy(update=dict(endpoint_key=endpoint_key))


LCP = LLMCallProps()
ResourceId = str
FileId = str
Attachments = list[list[ResourceId]]
Role = Literal["system", "assistant", "user"]


class Message(BaseModel, frozen=True):
    role: Role
    content: str

    @staticmethod
    def create(message: ChatMessage) -> "Message | None":
        return _create_message(message=message)

    def get_content(self):
        return self.content


def _create_message(message: ChatMessage) -> Message | None:
    msg = cast(BaseMessage, message)
    role: Role | None = "assistant" if msg.is_ai else "user" if msg.is_human else None
    return Message(role=role, content=msg.text) if role else None


class Messages(BaseModel, frozen=True):
    messages: list[Message]


class LLMPayload(Messages, frozen=True):
    attachments: Attachments | None = None

    def with_attachments(self, attachments: Attachments) -> "LLMPayload":
        return self.model_copy(update=dict(attachments=attachments))

    def __repr__(self):
        return self.show_pretty()

    def show_pretty(self, detailed: bool=False):
        total_size = sum(len(msg.content) for msg in self.messages)
        parts = [
            f"messages: {len(self.messages)}",
            f"total size: {total_size}" if detailed else None,
            "has attachments" if self.attachments else None,
        ]
        payload_pretty = ", ".join(filter(None, parts))
        return f"LLMPayload({payload_pretty})"

    @staticmethod
    def create(user_text: str, resource_id: ResourceId = "") -> "LLMPayload":
        return _create_payload(user_text=user_text, resource_id=resource_id)

    @staticmethod
    def parse(request: "Request") -> "LLMPayload":
        return _parse_payload(request)

    def get_resource_id(self) -> str | None:
        if not self.attachments:
            return None
        resource_id = self.attachments[0][0]
        return resource_id



def _create_payload(user_text: str, resource_id: ResourceId = ""):
    payload = LLMPayload(messages=[Message(role="user", content=user_text)])
    if not resource_id:
        return payload
    else:
        return payload.with_attachments(attachments=[[resource_id]])


class LLMResponseExt(BaseModel):
    text: str
    resource_id: ResourceId | None = None


RESPONSE_EMPTY = LLMResponseExt(text="")
LLMRequest = str | list[Message] | LLMPayload

def _parse_payload(request: LLMRequest) -> LLMPayload:
    if isinstance(request, str):
        return LLMPayload(messages=[Message(role="user", content=request)])
    elif isinstance(request, list) and all(isinstance(msg, Message) for msg in request):
        return LLMPayload(messages=request)
    elif isinstance(request, list) and all(isinstance(msg, dict) for msg in request):
        # todo validate contents of `msg`
        return LLMPayload(messages=request)
    elif isinstance(request, LLMPayload):
        return request
    elif isinstance(request, dict):
        # todo validate request fields
        return LLMPayload.model_validate(request)
    else:
        raise ValueError(f"Bad request type {type(request)}: {request}")


class LLMEndpointMetadata(BaseModel):
    key: str
    caption: str


class LLMHubMetadata(BaseModel):
    endpoints: list[LLMEndpointMetadata]
    default_endpoint_key: str

    def get_endpoint_keys(self):
        return [ep.key for ep in self.endpoints]

class LLMHubAPI:
    def get_metadata(self) -> LLMHubMetadata:
        raise NotImplementedError

    def get_response(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> str:
        raise NotImplementedError

    def get_response_ext(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> LLMResponseExt:
        raise NotImplementedError

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        raise NotImplementedError

# will be removed in the future
Request = LLMRequest
ResponseExt = LLMResponseExt
LLMAccessorAPI = LLMHubAPI
