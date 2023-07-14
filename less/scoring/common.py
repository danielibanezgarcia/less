from abc import ABC, abstractmethod
from typing import Iterable, List


class ScoringResult:
    def __init__(self, scores: Iterable, values: Iterable, found: Iterable, problem: Iterable):
        self._scores = list(scores)
        self._values = list(values)
        self._found = list(found)
        self._problem = list(problem)
        assert len(self._scores) == len(self._values) == len(self._found) == len(self._problem), "Scores, values, found and problem lists should have the same length"

    @property
    def values(self) -> list:
        return list(self._values)

    @property
    def scores(self) -> List[float]:
        return list(self._scores)

    @property
    def found(self) -> List[bool]:
        return list(self._found)

    @property
    def problem(self) -> List[bool]:
        return list(self._problem)


class WordFeatureScorer(ABC):
    def __init__(self, feature_name: str):
        self._feature_name = feature_name

    @property
    def feature(self) -> str:
        return self._feature_name

    @abstractmethod
    def score(self, candidates: List[str], target_word: str, context_before_word: List[str], context_after_word: List[str]) -> ScoringResult:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"feature: {self._feature_name}"
