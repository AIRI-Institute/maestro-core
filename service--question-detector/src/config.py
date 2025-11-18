import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mmar_utils import ExistingFile


class ModelsConfig(BaseModel):
    classifier: ExistingFile = "./src/data/classifier.jbl"
    first_word_list_path: ExistingFile = "./src/data/first_word_list.csv"

    basic_check_prompt: str = str(
        "Определи, является ли сообщение: <сообщение>{text}</сообщение>\n"
        "вопросом.\n"
        "Отвечай четко, в одно слово: Да или Нет"
    )


class LLMConfig(BaseModel):
    address: str = "llm-accessor:40631"
    max_retries: int = 3
    entrypoint_key: str = "giga-max-sberai"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(env_file=None):
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
