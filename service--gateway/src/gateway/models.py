import json
from collections.abc import Sequence
from typing import Any

from mmar_mapi import AIMessage, ChatMessage, DomainInfo, HumanMessage, ResourceId, TrackInfo
from pydantic import BaseModel, ConfigDict, Field, model_validator

FileName = str
FileData = tuple[FileName, bytes]
NamedResourceId = tuple[FileName, ResourceId]


class Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    @model_validator(mode="before")  # noqa
    @classmethod
    def validate_to_json(cls, value: str | Any) -> Any:
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class DBChatInfoItem(Base):
    chat_id: str = Field(alias="ChatId")
    first_replica: str | None = Field(default=None, alias="FirstReplica")
    first_replica_date: str | None = Field(default=None, alias="FirstReplicaDate")
    track_id: str = Field(alias="TrackId")


class DBChatPreviews(Base):
    chat_previews: Sequence[DBChatInfoItem] = Field(alias="ChatPreviews")


class EntrypointInfo(Base):
    entrypoint_key: str = Field(alias="EntrypointKey")
    caption: str = Field(alias="Caption")


class DomainsResponse(Base):
    domains: list[DomainInfo] = Field(alias="Domains")


class TracksResponse(Base):
    tracks: list[TrackInfo] = Field(alias="Tracks")


class EntrypointsResponse(Base):
    entrypoints: list[EntrypointInfo] = Field(alias="Entrypoints")
    default_entrypoint_key: str = Field(alias="DefaultEntrypointKey")


class UploadResponse(Base):
    resource_id: ResourceId = Field(alias="ResourceId")
    resource_name: str | None = Field(default=None, alias="ResourceName")
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
