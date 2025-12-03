from pathlib import Path
from loguru import logger
from enum import StrEnum
import ujson as json

from mmar_mapi.services import BinaryClassifiersAPI
from mmar_flame import Moderator

from src.config import Config


class Classifiers(StrEnum):
    black = "black"
    child = "child"
    greet = "greet"
    white = "white"
    receipt = "receipt"


def load_keywords(classifier: Classifiers) -> set[str]:
    logger.debug(f"Loading classifier: {classifier}")
    kws: set[str] = set(json.loads(Path(f"src/data/{classifier}list.json").read_text()))
    logger.debug(f"Loading classifier: {classifier}, keywords count: {len(kws)}")
    return kws


class Moderators(BinaryClassifiersAPI):
    def __init__(self, config: Config) -> None:
        self.moderators: dict[str, Moderator] = {clf: Moderator(keywords=load_keywords(clf)) for clf in Classifiers}

    def get_classifiers(self) -> list[str]:
        return ['moderator']

    def evaluate(self, *, classifier: str | None = None, text: str) -> bool:
        logger.debug(f"Moderator called, label: {classifier}")
        return self.moderators[classifier].evaluate(text=text)
