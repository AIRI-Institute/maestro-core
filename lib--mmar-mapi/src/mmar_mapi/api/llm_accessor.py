from typing import Literal

from pydantic import BaseModel

from mmar_mapi import ChatMessage


class EntrypointInfo(BaseModel):
    entrypoint_key: str
    caption: str


class EntrypointsConfig(BaseModel):
    entrypoints: list[EntrypointInfo]
    default_entrypoint_key: str


class LLMCallProps(BaseModel, frozen=True):
    entrypoint_key: str = ""
    attempts: int = 1


LCP = LLMCallProps()
ResourceId = str
FileId = str
Attachments = list[list[ResourceId]]


class Message(BaseModel, frozen=True):
    role: Literal["system", "assistant", "user"]
    content: str

    @staticmethod
    def create(message: ChatMessage) -> "Message":
        return _create_message(message=message)

    def get_content(self):
        return self.content


def _create_message(message: ChatMessage) -> Message | None:
    role = "assistant" if message.is_ai else "user" if message.is_human else None
    return Message(role=role, content=message.text) if role else None


class Messages(BaseModel, frozen=True):
    messages: list[Message]


class Payload(Messages, frozen=True):
    attachments: Attachments | None = None

    def with_attachments(self, attachments: Attachments) -> "Payload":
        return self.model_copy(update=dict(attachments=attachments))

    def __repr__(self):
        return self.show_pretty()

    def show_pretty(self, detailed: bool=False):
        total_size = sum(len(msg.content) for msg in self.messages)
        parts = [
            f"messages: {len(self.messages)}",
            f"total size: {total_size}" if detailed else None,
            self.attachments and "has attachments",
        ]
        payload_pretty = ", ".join(filter(None, parts))
        return f"Payload({payload_pretty})"        

    @staticmethod
    def create(user_text: str, resource_id: ResourceId = "") -> "Payload":
        return _create_payload(user_text=user_text, resource_id=resource_id)

    @staticmethod
    def parse(request: "Request") -> "Payload":
        return _parse_payload(request)

    def get_resource_id(self) -> str | None:
        if not self.attachments:
            return None
        resource_id = self.attachments[0][0]
        return resource_id


def _create_payload(user_text: str, resource_id: ResourceId = ""):
    payload = Payload(messages=[Message(role="user", content=user_text)])
    if not resource_id:
        return payload
    else:
        return payload.with_attachments(attachments=[[resource_id]])


class ResponseExt(BaseModel):
    text: str
    resource_id: ResourceId | None = None


RESPONSE_EMPTY = ResponseExt(text="")
Request = str | Messages | Payload


def _parse_payload(request: Request) -> Payload:
    if isinstance(request, str):
        return Payload(messages=[Message(role="user", content=request)])
    elif isinstance(request, list) and all(isinstance(msg, Message) for msg in request):
        return Payload(messages=request)
    elif isinstance(request, Payload):
        return request
    else:
        raise ValueError(f"Bad request type {type(request)}: {request}")


class LLMAccessorAPI:
    def get_entrypoints_config(self) -> EntrypointsConfig:
        raise NotImplementedError

    def get_entrypoint_keys(self) -> list[str]:
        raise NotImplementedError

    def get_response(self, *, request: Request, props: LLMCallProps = LCP) -> str:
        raise NotImplementedError

    def get_response_ext(self, *, request: Request, props: LLMCallProps = LCP) -> ResponseExt:
        raise NotImplementedError

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        raise NotImplementedError
