from dishka import Provider, Scope, provide
from mmar_mapi.services import LLMHubAPI
from mmar_ptag import ptag_client

from llm_hub_monitoring.config import Config, LangfuseConfig
from llm_hub_monitoring.llm_hub_monitoring import LLMHubMonitoring


class IOC(Provider):
    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return Config.load()  # type: ignore[return-value]

    @provide
    def langfuse_config(self, config: Config) -> LangfuseConfig:
        return config.langfuse

    @provide
    def llm_hub(self, config: Config) -> LLMHubAPI:
        llm_hub = ptag_client(LLMHubAPI, config.addresses.llm_hub_real)
        # llm_hub = LLMHubDummy()
        return llm_hub

    @provide
    def llm_hub_monitoring(self, llm_hub: LLMHubAPI, langfuse_config: LangfuseConfig) -> LLMHubMonitoring:
        return LLMHubMonitoring(llm_hub, langfuse_config)
