from abc import ABC, abstractmethod

from nltk import PorterStemmer, SnowballStemmer


class Stemmer(ABC):
    @abstractmethod
    def stem(self, word: str) -> str:
        raise NotImplementedError


class EnglishStemmer(Stemmer):
    def __init__(self):
        self._stemmer = PorterStemmer()

    def stem(self, word: str) -> str:
        return self._stemmer.stem(word)


class SpanishStemmer(Stemmer):
    def __init__(self):
        self._stemmer = SnowballStemmer('spanish')

    def stem(self, word: str) -> str:
        return self._stemmer.stem(word)
