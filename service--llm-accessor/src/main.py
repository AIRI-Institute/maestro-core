from mmar_ptag import deploy_server

from src.config import load_config
from src.config_server import load_config_server
from src.llm_accessor import LLMAccessor


def main():
    deploy_server(
        service=LLMAccessor,
        config=load_config,
        config_server=load_config_server,
    )


if __name__ == "__main__":
    main()
