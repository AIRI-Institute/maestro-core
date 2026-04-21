"""IoC container for dependency injection using dishka."""

import os
from pathlib import Path

from dishka import Provider, Scope, provide

from llm_hub.config import Config
from llm_hub.config_server import ConfigServer
from llm_hub.llm_hub import LLMHub


class IOCLocal(Provider):
    """IoC provider for application dependencies."""

    scope = Scope.APP

    @provide
    def get_config(self) -> Config:
        """Load and provide application configuration."""
        return Config.load()

    @provide
    def get_config_server(self) -> ConfigServer:
        return ConfigServer.load()

    @provide
    def get_llm_client(self, config: Config) -> LLMHub:
        """Create and provide LLM Hub OpenAI client."""
        return LLMHub(config.llm_config)
