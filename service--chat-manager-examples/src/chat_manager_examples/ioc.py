from dishka import Provider, Scope, provide
from mmar_mapi import FileStorage
from mmar_mapi.services import BinaryClassifiersAPI, LLMHubAPI, TextExtractorAPI, DocumentExtractorAPI
from mmar_mapi.tracks import TrackI, load_tracks
from mmar_ptag import ptag_client

from chat_manager_examples.chat_manager_examples import ChatManagerExamples
from chat_manager_examples.config import DOMAINS_CAPTIONS, TRACKS_MODULE, Config
from chat_manager_examples.legacy import provide_all_and_memoize


class IOC(Provider):
    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return Config.load()

    @provide
    def file_storage(self, config: Config) -> FileStorage:
        return FileStorage(config.files_dir)

    @provide
    def llm_hub(self, config: Config) -> LLMHubAPI:
        return ptag_client(LLMHubAPI, config.addresses.llm_hub)

    @provide
    def question_detector(self, config: Config) -> BinaryClassifiersAPI:
        return ptag_client(BinaryClassifiersAPI, config.addresses.question_detector)

    @provide
    def text_extractor(self, config: Config) -> TextExtractorAPI:
        return ptag_client(TextExtractorAPI, config.addresses.text_extractor)

    @provide
    def document_extractor(self, config: Config) -> DocumentExtractorAPI:
        return ptag_client(DocumentExtractorAPI, config.addresses.document_extractor)

    tracks = provide_all_and_memoize(
        provides=load_tracks(TRACKS_MODULE).values(),
        provides_result=list[TrackI],
    )

    @provide
    def chat_manager_examples(self, config: Config, tracks: list[TrackI]) -> ChatManagerExamples:
        no_such_track_text = config.messages.error.no_such_track_text
        return ChatManagerExamples(tracks, DOMAINS_CAPTIONS, no_such_track_text)
