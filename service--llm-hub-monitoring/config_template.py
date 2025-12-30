import os

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from mmar_mimpl import SettingsModel


class AddressesConfig(BaseModel):
    # llm_hub: '

class Config(SettingsModel):
    addresses_llm_hub
    ...
