from mmar_ptag import deploy_server

from question_detector.question_detector import QuestionDetector
from question_detector.config import load_config
from question_detector.config_server import load_config_server


def main():
    deploy_server(
        service=QuestionDetector,
        config=load_config,
        config_server=load_config_server,
    )


if __name__ == "__main__":
    main()
