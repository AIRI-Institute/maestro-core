from loguru import logger
from mmar_mapi.services import BinaryClassifiersAPI
from mmar_flame import Moderator


class Moderators(BinaryClassifiersAPI):
    def __init__(self, moderators: dict[str, Moderator]) -> None:
        self.moderators = moderators

    def get_classifiers(self) -> list[str]:
        return list(self.moderators.keys())

    def evaluate(self, *, classifier: str | None = None, text: str) -> bool:
        classifier = classifier or "black"

        if classifier not in self.moderators:
            raise ValueError(f"Classifier '{classifier}' not found")

        logger.trace(f"Moderator called, label: {classifier}")
        return self.moderators[classifier].evaluate(text=text)
