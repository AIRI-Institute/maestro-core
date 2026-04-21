from pathlib import Path

import ujson as json
from dishka import Provider, Scope, provide
from mmar_flame import Moderator

from moderators.config import Config, load_config
from moderators.moderators import Moderators

CLF = [
    "black",
    "child",
    "greet",
    "white",
    "receipt",
]


class IOC(Provider):
    scope = Scope.APP


class IOCLocal(Provider):
    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return load_config()

    @provide
    def moderators_map(self) -> dict[str, Moderator]:
        data_dir = Path(__file__).parent / "data"
        return {clf: Moderator(keywords=set(json.loads((data_dir / f"{clf}list.json").read_text()))) for clf in CLF}

    @provide
    def moderators(self, moderators_map: dict[str, Moderator]) -> Moderators:
        return Moderators(moderators_map)


IOCS = [IOC, IOCLocal]
