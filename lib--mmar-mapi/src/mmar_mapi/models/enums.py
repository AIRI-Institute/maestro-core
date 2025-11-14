from enum import StrEnum, auto


class DoctorChoiceXMLTagEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return f"{name.lower()}"

    DIAGNOSTICS = auto()
    SUMMARIZATION = auto()
    MTRS = auto()


class MTRSXMLTagEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return f"mtrs_{name.lower()}"

    NAME = auto()
    LABEL = auto()
    DESC = auto()


class MTRSLabelEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name.upper()

    LABORATORY = auto()
    INSTRUMENTAL = auto()


class DiagnosticsXMLTagEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return f"diag_{name.lower()}"

    DIAG = auto()
    DOC = auto()
    DESC = auto()
