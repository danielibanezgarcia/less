from typing import List

from wordfreq import zipf_frequency

from less.common.util import normalize
from less.scoring.common import ScoringResult, WordFeatureScorer


class FrequencyScorer(WordFeatureScorer):
    def __init__(self, lang: str, min_allowed_zipf_freq: float = None):
        super().__init__("Frequency")
        self._lang = lang
        self._min_allowed_zipf_freq = min_allowed_zipf_freq or 1.0

    def score(self, candidates: List[str], target_word: str, context_before_word: List[str], context_after_word: List[str]) -> ScoringResult:
        values = [self._zipf_freq(candidate) for candidate in candidates]
        scores = [normalize(value, 1, 7) for value in values]
        found = [value > 0 for value in values]
        problem = [candidate_freq < self._min_allowed_zipf_freq for candidate_freq in values]
        return ScoringResult(scores, values, found, problem)

    def _zipf_freq(self, word: str) -> float:
        zipf_freq = zipf_frequency(word, self._lang)
        if zipf_freq == 0 and word != word.lower():
            zipf_freq = zipf_frequency(word.lower(), self._lang)
        return zipf_freq
