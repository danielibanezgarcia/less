import pickle
from os import path
from typing import List

from less.common.util import get_ngrams, get_window_context
from less.scoring.common import ScoringResult, WordFeatureScorer


class WordFluencyScorer(WordFeatureScorer):
    def __init__(self, lang: str, ngrams_dir: str, max_ngram_size: int = 2, min_ngram_size: int = 2):
        super().__init__("Fluency")
        self._lang = lang
        self._ngram_sizes = list(range(min_ngram_size, max_ngram_size + 1))
        bigrams_file_path = path.join(ngrams_dir, f"{lang}-2-grams.pkl")
        with open(bigrams_file_path, "rb") as bigrams_file:
            self._bigram_freq = pickle.load(bigrams_file)

    def score(self, candidates: List[str], target_word: str, context_before_word: List[str], context_after_word: List[str]) -> ScoringResult:
        scores = [self._score_candidate(candidate, context_before_word, context_after_word) for candidate in candidates]
        found = [True] * len(scores)
        problem = [False] * len(scores)
        return ScoringResult(scores, scores, found, problem)

    def _score_candidate(self, candidate: str, context_before_word: List[str], context_after_word: List[str]):
        ngrams = []
        for ngram_size in self._ngram_sizes:
            left_context_words, right_context_words = get_window_context(context_before_word, context_after_word, ngram_size)
            candidate_in_context = left_context_words + [candidate] + right_context_words
            ngrams.extend(get_ngrams(candidate_in_context, ngram_size))
        scores = [self._score_ngram(ngram) for ngram in ngrams]
        return sum(scores) / len(scores)

    def _score_ngram(self, ngram: List[str]) -> float:
        ngram_text = " ".join(ngram)
        value = self._bigram_freq.get(ngram_text, 0)
        return value
