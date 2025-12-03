from loguru import logger
from mmar_mapi import AIMessage, Chat, ChatMessage, DomainInfo, TrackInfo
from mmar_mapi.services import ChatManagerAPI
from mmar_mapi.tracks import TrackI
from mmar_utils import pretty_line


class ChatManagerExamples(ChatManagerAPI):
    def __init__(self, tracks: list[TrackI], domains_captions: dict[str, str], no_such_track_text: str) -> None:
        self.tracks_map = {type(tr).__name__: tr for tr in tracks}
        self.domains_captions = domains_captions
        track_ids_pretty = ", ".join(self.tracks_map.keys())
        logger.info(f"Tracks loaded: [{track_ids_pretty}]")
        self.no_such_track = AIMessage(content=no_such_track_text)

    def get_domains(self, *, client_id: str, language_code: str = "ru") -> list[DomainInfo]:
        domains = [DomainInfo(domain_id=did, name=name) for did, name in self.domains_captions.items()]
        logger.info(f"#get_domains({language_code=}, {client_id=}) -> {domains}")
        return domains

    def get_tracks(self, *, client_id: str, language_code: str = "ru") -> list[TrackInfo]:
        tracks = [TrackInfo(track_id=tid, name=tr.CAPTION, domain_id=tr.DOMAIN) for tid, tr in self.tracks_map.items()]
        logger.info(f"#get_tracks({client_id=}, {language_code=}) -> {tracks}")
        return tracks

    def get_response(self, *, chat: Chat) -> list[ChatMessage]:
        chat_id = chat.create_id()
        logger.debug(f"Processing {chat_id}: started, language_code={chat.context.language_code}")
        track_id = chat.context.track_id
        track: TrackI | None = self.tracks_map.get(track_id)
        if track is None:
            logger.error(f"Track with track_id=`{track_id}` is not present!")
            return [self.no_such_track.with_now_datetime()]

        messages = track.get_response(chat=chat)

        logger.debug(f"Processing {chat_id} -> {pretty_line(repr(messages), cut_count=400)}")
        return messages
