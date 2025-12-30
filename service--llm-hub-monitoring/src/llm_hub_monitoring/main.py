from dishka import make_container
from mmar_ptag import deploy_server

from llm_hub_monitoring.config import Config
from llm_hub_monitoring.config_server import ConfigServer
from llm_hub_monitoring.ioc import IOC
from llm_hub_monitoring.llm_hub_monitoring import LLMHubMonitoring


def main():
    container = make_container(IOC())
    service = container.get(LLMHubMonitoring)
    config = container.get(Config)
    deploy_server(
        service=service,
        config=config,
        config_server=ConfigServer.load,
    )


if __name__ == "__main__":
    main()
