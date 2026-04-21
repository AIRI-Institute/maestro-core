from dishka import make_container
from mmar_ptag import deploy_server

from question_detector.config import Config
from question_detector.config_server import ConfigServer
from question_detector.ioc import IOCS
from question_detector.question_detector import QuestionDetector


def main():
    container = make_container(*[ioc() for ioc in IOCS])
    config = container.get(Config)
    service = container.get(QuestionDetector)
    deploy_server(
        service=service,
        config=config,
        config_server=ConfigServer,
    )


if __name__ == "__main__":
    main()
