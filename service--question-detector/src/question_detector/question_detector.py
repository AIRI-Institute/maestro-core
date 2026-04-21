import numpy as np
from mmar_mapi.services import BinaryClassifiersAPI
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

from question_detector.config import Config
from question_detector.models import FirstWordCheck, GigaBasicCheck, GigaEmbedding, QuestionMarkCheck

QUESTION_DETECTOR = "question-detector"


class QuestionDetector(BinaryClassifiersAPI):
    def __init__(
        self,
        giga_embedding: GigaEmbedding,
        question_mark: QuestionMarkCheck,
        giga_basic: GigaBasicCheck,
        first_word: FirstWordCheck,
        classifier: HistGradientBoostingClassifier,
        config: Config,
    ):
        self.giga_embedding = giga_embedding
        self.question_mark = question_mark
        self.giga_basic = giga_basic
        self.first_word = first_word
        self.classifier: HistGradientBoostingClassifier = classifier
        self.config = config

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
        return self.classifier.predict(vector)

    def _get_vector(self, text: str) -> np.ndarray:
        embedding_v = self.giga_embedding(text)
        question_mark_v = self.question_mark(text)
        giga_basic_v = self.giga_basic(text)
        first_word_v = self.first_word(text)
        vector = np.concat([embedding_v, question_mark_v, giga_basic_v, first_word_v], axis=0).reshape(1, -1)

        # Validate vector dimension matches classifier expectations
        expected_features = self.classifier.n_features_in_
        actual_features = vector.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: classifier expects {expected_features} features "
                f"but got {actual_features}. "
                f"Embedding dim: {embedding_v.shape[0]} (expected: {self.config.models.expected_embedding_dim}), "
                f"QuestionMark dim: {question_mark_v.shape[0]}, "
                f"GigaBasic dim: {giga_basic_v.shape[0]}, "
                f"FirstWord dim: {first_word_v.shape[0]}. "
                f"Check the embedding model configuration."
            )
        return vector
