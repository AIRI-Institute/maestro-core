import joblib as jbl
import numpy as np
from mmar_mapi.services import BinaryClassifiersAPI, LLMHubAPI
from mmar_ptag import ptag_client
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

from question_detector.config import Config
from question_detector.models import FirstWordCheck, GigaBasicCheck, GigaEmbedding, QuestionMarkCheck

QUESTION_DETECTOR = "question-detector"


class QuestionDetector(BinaryClassifiersAPI):
    def __init__(self, config: Config):
        llm_hub = ptag_client(LLMHubAPI, config.llm.address)

        self.giga_embedding = GigaEmbedding(config, llm_hub)
        self.question_mark = QuestionMarkCheck()
        self.giga_basic = GigaBasicCheck(config, llm_hub)
        self.first_word = FirstWordCheck(config.models)
        self.clf: HistGradientBoostingClassifier = jbl.load(config.models.classifier)

    def get_classifiers(self) -> list[str]:
        return [QUESTION_DETECTOR]

    def evaluate(self, *, classifier: str | None = None, text: str) -> bool:
        if classifier is not None and classifier != QUESTION_DETECTOR:
            raise ValueError(f"Only classifier={QUESTION_DETECTOR} supported, found: {classifier}")
        vector: np.ndarray = self._get_vector(text)
        prediction = bool(int(self._predict(vector)[0]))
        logger.debug(f"Evaluating text: {text} -> {prediction}")
        return prediction

    def _predict(self, vector: np.ndarray):
        return self.clf.predict(vector)

    def _get_vector(self, text: str) -> np.ndarray:
        embedding_v = self.giga_embedding(text)
        question_mark_v = self.question_mark(text)
        giga_basic_v = self.giga_basic(text)
        first_word_v = self.first_word(text)
        return np.concat([embedding_v, question_mark_v, giga_basic_v, first_word_v], axis=0).reshape(1, -1)
