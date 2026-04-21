from pydantic import BaseModel

from mmar_mapi.models.chat import Chat


Interpretation = str
ResourceId = str


class ContentInterpreterRemoteResponse(BaseModel):
    interpretation: str
    resource_fname: str
    resource: bytes


class ContentInterpreterRemoteAPI:
    def interpret_remote(
        self, *, kind: str, query: str, resource: bytes, chat: Chat | None = None
    ) -> ContentInterpreterRemoteResponse:
        raise NotImplementedError


class ContentInterpreterAPI:
    def interpret(
        self, *, kind: str, query: str, resource_id: str = "", chat: Chat | None = None
    ) -> tuple[Interpretation, ResourceId | None]:
        raise NotImplementedError
