import json
from collections.abc import Sequence
from typing import Any

from mmar_mapi import AIMessage, ChatMessage, DomainInfo, HumanMessage, ResourceId, TrackInfo
from pydantic import BaseModel, ConfigDict, model_validator

FileName = str
FileData = tuple[FileName, bytes]
NamedResourceId = tuple[FileName, ResourceId]


class Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="before")  # noqa
    @classmethod
    def validate_to_json(cls, value: str | Any) -> Any:
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class DBChatInfoItem(Base):
    chat_id: str
    first_replica: str | None = None
    first_replica_date: str | None = None
    track_id: str


class DBChatPreviews(Base):
    chat_previews: Sequence[DBChatInfoItem]


class DomainsResponse(Base):
    domains: list[DomainInfo]


class TracksResponse(Base):
    tracks: list[TrackInfo]


class ModelInfo(Base):
    model: str
    caption: str


class ModelsResponse(Base):
    models: list[ModelInfo]
    default_model: str


class UploadResponse(Base):
    resource_id: ResourceId
    resource_name: str | None = None
    error: str | None = None


class UploadManyRequest(Base):
    resource_ids: Sequence[ResourceId]


class CreateResponse(BaseModel):
    chat_id: str


class ChatRequest(BaseModel):
    chat_id: str
    messages: list[HumanMessage]


class ChatRequestMessages(BaseModel):
    messages: list[HumanMessage]


class ChatResponse(BaseModel):
    chat_id: str
    response_messages: list[AIMessage]


class HistoryRequest(BaseModel):
    chat_id: str


class HistoryResponse(HistoryRequest):
    chat_id: str
    messages: list[ChatMessage]
