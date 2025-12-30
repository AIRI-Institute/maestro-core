from pydantic import BaseModel, Field

from mmar_mimpl import SettingsModel


class AddressesConfig(BaseModel):
    # this is bad naming: probably it's better to use the same idiom as in chat-manager
    # i.e.
    # addresses__llm_hub for real llm-hub
    # addresses__llm_hub_facade everywhere, and it can be the same
    # i.e. llm_hub_facade for all, and if we want, it's the same as for 
    llm_hub_real: str = "llm-hub:40631"


class LangfuseConfig(BaseModel):
    enabled: bool = Field(default=True, alias="LANGFUSE_ENABLED")
    public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    host: str = Field(default="", alias="LANGFUSE_HOST")
    capture_content: bool = Field(default=True, alias="LANGFUSE_CAPTURE_CONTENT")

    class Config:
        populate_by_name = True


class Config(SettingsModel):
    addresses: AddressesConfig = AddressesConfig()
    langfuse: LangfuseConfig = LangfuseConfig()
