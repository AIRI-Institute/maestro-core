import os
from pathlib import Path

from mmar_utils import ExistingFile
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from mmar_mimpl import SettingsModel

DATA_DIR = Path(__file__).parent / "data"

# Expected embedding dimension for the GigaEmbedding model
# The classifier was trained with this specific dimension
EXPECTED_EMBEDDING_DIM = 1024


class ModelsConfig(BaseModel):
    classifier: ExistingFile = f"{DATA_DIR}/classifier.jbl"
    first_word_list_path: ExistingFile = f"{DATA_DIR}/first_word_list.csv"

    # Expected embedding dimension for the embedding model
    # The classifier was trained expecting this specific dimension
    expected_embedding_dim: int = EXPECTED_EMBEDDING_DIM

    basic_check_prompt: str = str(
        "Определи, является ли сообщение: <сообщение>{text}</сообщение>\n"
        "вопросом.\n"
        "Отвечай четко, в одно слово: Да или Нет"
    )


class AddressesConfig(BaseModel):
    llm_hub: str = "llm-hub:40631"

class LLMConfig(BaseModel):
    max_retries: int = 3
    question_detector_model: str = ""


class Config(SettingsModel):
    addresses: AddressesConfig = AddressesConfig()
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
