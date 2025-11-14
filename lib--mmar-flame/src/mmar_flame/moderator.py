from pathlib import Path
import ujson as json
import re
import pymorphy3
from nltk.util import ngrams as nltk_ngrams
from transliterate import translit


class Moderator:
    def __init__(self, *, keywords: set[str], lang: str = "ru", ngrams_sizes: tuple[int] = (-1, 1, 2, 3)) -> None:
        self.keywords: set[str] = keywords
        self.lang: str = lang
        self.morph = pymorphy3.MorphAnalyzer(lang=lang)
        self.ngrams_sizes: tuple[int] = ngrams_sizes

    def evaluate(self, *, text: str) -> bool:
        n_grams = {ngram_size: self._text_to_n_grams(text, n=ngram_size) for ngram_size in self.ngrams_sizes}
        if self.check(self.keywords, n_grams, max_ngram_size=3):
            return True
        return False

    def _text_to_n_grams(self, input_text: str, n: int =-1) -> list[str]:
        """n=-1 is for full text"""
        text = input_text.lower()
        text = translit(text, self.lang)
        text = re.sub(r"[^a-zа-я0-9_]+", " ", text).strip()
        words = [self.morph.parse(word)[0].normal_form for word in text.split()]
        if n == -1:
            return [" ".join(words)]
        return [" ".join(ngram) for ngram in nltk_ngrams(words, n)]

    @staticmethod
    def check(keywords: set[str], n_grams: dict[int, list[str]], max_ngram_size: int = 3) -> bool:
        for ngram_size, divided_text in n_grams.items():
            if ngram_size > max_ngram_size:
                return False
            if any(ngram in keywords for ngram in divided_text):
                return True
        return False

    @staticmethod
    def load_keywords(keywords_path: Path) -> set[str]:
        return set(json.loads(keywords_path.read_text()))