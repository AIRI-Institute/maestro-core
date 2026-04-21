from dishka import make_container
from mmar_ptag import deploy_server

from moderators.config import Config
from moderators.config_server import load_config_server
from moderators.ioc import IOCS
from moderators.moderators import Moderators


def main():
    container = make_container(*(ioc() for ioc in IOCS))
    config = container.get(Config)
    service = container.get(Moderators)

    deploy_server(
        service=service,
        config=config,
        config_server=load_config_server,
    )


if __name__ == "__main__":
    main()