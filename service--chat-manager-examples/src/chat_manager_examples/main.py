from dishka import make_container
from mmar_ptag import deploy_server

from chat_manager_examples.chat_manager_examples import ChatManagerExamples
from chat_manager_examples.ioc import IOC
from chat_manager_examples.config import Config
from chat_manager_examples.config_server import ConfigServer


def main():
    container = make_container(IOC())
    service = container.get(ChatManagerExamples)
    config = container.get(Config)
    deploy_server(
        service=service,
        config=config,
        config_server=ConfigServer.load,
    )


if __name__ == "__main__":
    main()
