class BinaryClassifiersAPI:
    def get_classifiers(self) -> list[str]:
        raise NotImplementedError

    def evaluate(self, *, classifier: str | None = None, text: str) -> bool:
        raise NotImplementedError
