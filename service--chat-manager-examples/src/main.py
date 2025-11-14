from mmar_ptag import deploy_server

from src.chat_manager_examples import ChatManagerExamples
from src.config import load_config
from src.config_server import load_config_server


def main():
    deploy_server(
        service=ChatManagerExamples,
        config=load_config,
        config_server=load_config_server,
    )


if __name__ == "__main__":
    main()
