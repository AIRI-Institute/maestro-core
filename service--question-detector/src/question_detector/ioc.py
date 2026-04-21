import joblib as jbl
from dishka import Provider, Scope, provide
from mmar_llm.legacy import fix_base_url
from question_detector.config import Config
from question_detector.models import FirstWordCheck, GigaBasicCheck, GigaEmbedding, QuestionMarkCheck
from question_detector.question_detector import QuestionDetector
from sklearn.ensemble import HistGradientBoostingClassifier
from openai import OpenAI


class IOC(Provider):
    scope = Scope.APP

    @provide
    def oclient(self, config: Config) -> OpenAI:
        base_url = fix_base_url(config.addresses.llm_hub)
        return OpenAI(
            base_url=base_url,
            api_key="_",
            max_retries=config.llm.max_retries,
        )


class IOCLocal(Provider):
    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return Config.load()

    @provide
    def classifier(self, config: Config) -> HistGradientBoostingClassifier:
        return jbl.load(config.models.classifier)

    @provide
    def question_detector(
        self,
        config: Config,
        oclient: OpenAI,
        classifier: HistGradientBoostingClassifier,
    ) -> QuestionDetector:
        giga_embedding = GigaEmbedding(config, oclient)
        question_mark = QuestionMarkCheck()
        giga_basic = GigaBasicCheck(config, oclient)
        first_word = FirstWordCheck(config.models)
        return QuestionDetector(
            giga_embedding=giga_embedding,
            question_mark=question_mark,
            giga_basic=giga_basic,
            first_word=first_word,
            classifier=classifier,
            config=config,
        )


IOCS = [IOC, IOCLocal]
