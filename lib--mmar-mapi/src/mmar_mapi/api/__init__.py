from pydantic import BaseModel

from mmar_mapi.api.document_extractor import (
    DocExtractionOutput,
    DocExtractionSpec,
    DocumentExtractorAPI,
    ExtractedImage,
    ExtractedImageMetadata,
    ExtractedPageImage,
    ExtractedPicture,
    ExtractedTable,
    ExtractionEngineSpec,
    ForceOCR,
    OutputType,
    PageRange,
    DOC_SPEC_DEFAULT,
)
from mmar_mapi.api.llm_accessor import (
    LCP,
    RESPONSE_EMPTY,
    Attachments,
    LLMAccessorAPI,
    LLMCallProps,
    Message,
    Messages,
    Payload,
    Request,
    ResponseExt,
    EntrypointInfo,
    EntrypointsConfig,
)
from mmar_mapi.models.chat import Chat, ChatMessage
from mmar_mapi.models.tracks import DomainInfo, TrackInfo

# variable to prevent removing unused imports
__imported__ = [
    # llm_accessor
    LLMCallProps,
    LCP,
    Attachments,
    Message,
    Messages,
    Payload,
    Request,
    ResponseExt,
    RESPONSE_EMPTY,
    LLMAccessorAPI,
    # document_extractor
    PageRange,
    ForceOCR,
    OutputType,
    ExtractionEngineSpec,
    DocExtractionSpec,
    ExtractedImage,
    ExtractedImageMetadata,
    ExtractedPicture,
    ExtractedTable,
    ExtractedPageImage,
    DocExtractionOutput,
    DocumentExtractorAPI,
    DOC_SPEC_DEFAULT,
    # entrypoints
    EntrypointInfo,
    EntrypointsConfig,
]


Interpretation = str
ResourceId = str


class ChatManagerAPI:
    def get_domains(self, *, client_id: str, language_code: str = "ru") -> list[DomainInfo]:
        raise NotImplementedError

    def get_tracks(self, *, client_id: str, language_code: str = "ru") -> list[TrackInfo]:
        raise NotImplementedError

    def get_response(self, *, chat: Chat) -> list[ChatMessage]:
        raise NotImplementedError


class TextGeneratorAPI:
    def process(self, *, chat: Chat) -> str:
        raise NotImplementedError


class ContentInterpreterRemoteResponse(BaseModel):
    interpretation: str
    resource_fname: str
    resource: bytes


class ContentInterpreterRemoteAPI:
    def interpret_remote(
        self, *, kind: str, query: str, resource: bytes, chat: Chat | None = None
    ) -> ContentInterpreterRemoteResponse:
        raise NotImplementedError


class BinaryClassifiersAPI:
    def get_classifiers(self) -> list[str]:
        raise NotImplementedError

    def evaluate(self, *, classifier: str | None = None, text: str) -> bool:
        raise NotImplementedError


class TranslatorAPI:
    def get_lang_codes(self) -> list[str]:
        raise NotImplementedError

    def translate(self, *, text: str, lang_code_from: str | None = None, lang_code_to: str) -> str:
        raise NotImplementedError


class CriticAPI:
    def evaluate(self, *, text: str, chat: Chat | None = None) -> float:  # TODO replace float with bool
        raise NotImplementedError


class ContentInterpreterAPI:
    def interpret(
        self, *, kind: str, query: str, resource_id: str = "", chat: Chat | None = None
    ) -> tuple[Interpretation, ResourceId | None]:
        raise NotImplementedError


class TextProcessorAPI:
    def process(self, *, text: str, chat: Chat | None = None) -> str:
        raise NotImplementedError


class TextExtractorAPI:
    def extract(self, *, resource_id: ResourceId) -> ResourceId:
        """returns file with text"""
        raise NotImplementedError
