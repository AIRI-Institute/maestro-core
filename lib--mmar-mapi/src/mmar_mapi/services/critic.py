from mmar_mapi.models.chat import Chat


class CriticAPI:
    def evaluate(self, *, text: str, chat: Chat | None = None) -> float:  # TODO replace float with bool
        raise NotImplementedError
