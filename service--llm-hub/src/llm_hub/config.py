from mmar_llm import LLMConfig, LLMHubConfig

from mmar_mimpl import LoadPydanticModel, SettingsModel


class Config(SettingsModel):
    files_dir: str = "/mnt/data/maestro/files"

    llm_config_path: str
    llm: LoadPydanticModel[LLMConfig, "llm_config_path"] = None
