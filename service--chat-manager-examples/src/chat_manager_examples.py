from loguru import logger
from mmar_mapi import AIMessage, Chat, ChatMessage, DomainInfo, TrackInfo
from mmar_mapi.api import ChatManagerAPI, LLMAccessorAPI
from mmar_mapi.tracks import TrackI, load_tracks
from mmar_ptag import ptag_client

from src.config import DOMAINS_CAPTIONS, TRACKS_MODULE, Config


class ChatManagerExamples(ChatManagerAPI):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm_accessor = ptag_client(LLMAccessorAPI, config.addresses.llm_accessor)

        track_classes = load_tracks(TRACKS_MODULE)
        self.tracks = {t_id: t_class(config) for t_id, t_class in track_classes.items()}
        track_ids_pretty = ", ".join(self.tracks.keys())
        logger.info(f"Tracks loaded: [{track_ids_pretty}]")
        self.no_such_track = AIMessage(content=self.config.messages.error.no_such_track)

    def get_domains(self, *, client_id: str, language_code: str = "ru") -> list[DomainInfo]:
        domains = [DomainInfo(domain_id=did, name=name) for did, name in DOMAINS_CAPTIONS.items()]
        logger.info(f"#get_domains({language_code=}, {client_id=}) -> {domains}")
        return domains

    def get_tracks(self, *, client_id: str, language_code: str = "ru") -> list[TrackInfo]:
        tracks = [TrackInfo(track_id=tid, name=tr.CAPTION, domain_id=tr.DOMAIN) for tid, tr in self.tracks.items()]
        logger.info(f"#get_tracks({client_id=}, {language_code=}) -> {tracks}")
        return tracks

    def get_response(self, *, chat: Chat) -> list[ChatMessage]:
        chat_id = chat.create_id()
        logger.debug(f"Processing {chat_id}: started, language_code={chat.context.language_code}")
        track_id = chat.context.track_id
        track: TrackI | None = self.tracks.get(track_id)
        if track is None:
            logger.error(f"Track with track_id=`{track_id}` is not present!")
            return [self.no_such_track.with_now_datetime()]

        messages = track.get_response(chat=chat)

        logger.debug(f"Processing {chat_id} -> {messages}")
        return messages
