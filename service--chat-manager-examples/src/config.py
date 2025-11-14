import os
from pathlib import Path
from types import SimpleNamespace

from mmar_ptag import LogLevelEnum
from mmar_utils import ExistingFile
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TRACKS_MODULE = "src.tracks"
DOMAINS_ALL = dict(examples=("Examples", "Примеры"))
DOMAINS = SimpleNamespace(**{k: v[0] for k, v in DOMAINS_ALL.items()})
DOMAINS_CAPTIONS = {v[0]: v[1] for v in DOMAINS_ALL.values()}
CLIENTS_KEY = "CLIENTS"


class AddressesConfig(BaseModel):
    moderator: str = "moderator:31111"
    text_extractor: str = "text-extractor:9681"
    llm_accessor: str = "llm-accessor:40631"


class MessagesErrorConfig(BaseModel):
    no_such_track: str = "Этот сценарий еще не проработан."


class MessagesConfig(BaseModel):
    error: MessagesErrorConfig = MessagesErrorConfig()


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"
    files_dir: str = Field(default="/mnt/data/maestro/files")

    messages: MessagesConfig = MessagesConfig()
    addresses: AddressesConfig = AddressesConfig()


def load_config(env_file=None):
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
