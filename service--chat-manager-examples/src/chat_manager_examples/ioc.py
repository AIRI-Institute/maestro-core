from dishka import Provider, Scope, provide
from mmar_mapi import FileStorage
from mmar_mapi.services import BinaryClassifiersAPI, DocumentExtractorAPI, TextExtractorAPI
from mmar_mapi.tracks import TrackI
from mmar_ptag import ptag_client
from openai import OpenAI

from chat_manager_examples.chat_manager_examples import ChatManagerExamples
from chat_manager_examples.config import DOMAINS_CAPTIONS, Config
from chat_manager_examples.tracks.chatbot import Chatbot
from chat_manager_examples.tracks.describer import Describer
from chat_manager_examples.tracks.document_describer import DocumentDescriber
from chat_manager_examples.tracks.dummy import Dummy
from chat_manager_examples.tracks.filling_user_profile import FillingUserProfile
from chat_manager_examples.tracks.llm_config_wizard import LLMConfigWizard
from chat_manager_examples.tracks.recipes_summarizer import RecipesSummarizer
from chat_manager_examples.tracks.simple import Simple


class IOC(Provider):
    scope = Scope.APP

    @provide
    def file_storage(self, config: Config) -> FileStorage:
        return FileStorage(config.files_dir)

    @provide
    def question_detector(self, config: Config) -> BinaryClassifiersAPI:
        return ptag_client(BinaryClassifiersAPI, config.addresses.question_detector)

    @provide
    def text_extractor(self, config: Config) -> TextExtractorAPI:
        return ptag_client(TextExtractorAPI, config.addresses.text_extractor)

    @provide
    def document_extractor(self, config: Config) -> DocumentExtractorAPI:
        return ptag_client(DocumentExtractorAPI, config.addresses.document_extractor)

    @provide
    def oclient(self, config: Config) -> OpenAI:
        return OpenAI(base_url=config.openai_api_base, api_key=config.openai_api_key)


class IOCLocal(Provider):
    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return Config.load()

    chatbot = provide(Chatbot)
    describer = provide(Describer)
    document_describer = provide(DocumentDescriber)
    dummy = provide(Dummy)
    filling_user_profile = provide(FillingUserProfile)
    llm_config_wizard = provide(LLMConfigWizard)
    recipes_summarizer = provide(RecipesSummarizer)
    simple = provide(Simple)

    @provide
    def tracks(
        self,
        chatbot: Chatbot,
        describer: Describer,
        document_describer: DocumentDescriber,
        dummy: Dummy,
        filling_user_profile: FillingUserProfile,
        llm_config_wizard: LLMConfigWizard,
        recipes_summarizer: RecipesSummarizer,
        simple: Simple,
    ) -> list[TrackI]:
        return [
            chatbot,
            describer,
            document_describer,
            dummy,
            filling_user_profile,
            llm_config_wizard,
            recipes_summarizer,
            simple,
        ]

    @provide
    def chat_manager_examples(self, config: Config, tracks: list[TrackI]) -> ChatManagerExamples:
        return ChatManagerExamples(tracks, DOMAINS_CAPTIONS, config.messages.error.no_such_track)


IOCS = [IOC, IOCLocal]
