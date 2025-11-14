import re
from enum import StrEnum


class XMLParser:
    @staticmethod
    def make_xml(body: str, tag: str | StrEnum) -> str:
        return f"<{tag}>{body}</{tag}>"

    @staticmethod
    def remove_xml(text: str) -> str:
        re_xml = re.compile(r"<(.*?)>")
        return re.sub(re_xml, "", text).replace("  ", " ").strip()

    @staticmethod
    def get_tag_list(response: str, tag: str | StrEnum) -> list[str]:
        tag_content = re.findall(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        return [content.lower().strip() for content in tag_content]
