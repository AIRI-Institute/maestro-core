from types import SimpleNamespace

from mmar_mimpl import SettingsModel
from pydantic import BaseModel, Field

TRACKS_MODULE = "chat_manager_examples.tracks"
DOMAINS_ALL = {"examples": ("Examples", "Примеры")}
DOMAINS = SimpleNamespace(**{k: v[0] for k, v in DOMAINS_ALL.items()})
DOMAINS_CAPTIONS = {v[0]: v[1] for v in DOMAINS_ALL.values()}
CLIENTS_KEY = "CLIENTS"


class AddressesConfig(BaseModel):
    moderator: str = "moderator:31111"
    text_extractor: str = "text-extractor:9681"
    llm_hub: str = Field(default="llm-hub:40631")
    question_detector: str = "question-detector:31611"
    document_extractor: str = "document-extractor:9671"


class MessagesErrorConfig(BaseModel):
    no_such_track: str = "Этот сценарий еще не проработан."


class MessagesConfig(BaseModel):
    error: MessagesErrorConfig = MessagesErrorConfig()


class Config(SettingsModel):
    files_dir: str = Field(default="/mnt/data/maestro/files")

    messages: MessagesConfig = MessagesConfig()
    addresses: AddressesConfig = AddressesConfig()

    openai_api_base: str = "http://llm-hub:40631/v1"
    openai_api_key: str = "_"
