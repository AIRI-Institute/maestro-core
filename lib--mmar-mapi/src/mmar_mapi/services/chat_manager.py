from mmar_mapi.models.chat import Chat, ChatMessage
from mmar_mapi.models.tracks import DomainInfo, TrackInfo


class ChatManagerAPI:
    def get_domains(self, *, client_id: str, language_code: str = "ru") -> list[DomainInfo]:
        raise NotImplementedError

    def get_tracks(self, *, client_id: str, language_code: str = "ru") -> list[TrackInfo]:
        raise NotImplementedError

    def get_response(self, *, chat: Chat) -> list[ChatMessage]:
        raise NotImplementedError
