from typing import List

from less.embedding.common import EmbeddingModel
from less.scoring.common import ScoringResult, WordFeatureScorer


class ContextSimilarityScorer(WordFeatureScorer):
    def __init__(self, model: EmbeddingModel, context_size: int = 5):
        if context_size % 2 == 0:
            raise ValueError(f"The context_window argument is {context_size} but it should be an odd number")
        super().__init__("Context Similarity")
        self._model = model
        self._max_context_words = (context_size - 1) // 2

    def score(self, candidates: List[str], target_word: str, context_before_word: List[str], context_after_word: List[str]) -> ScoringResult:
        context_words = self._get_context(context_before_word, context_after_word)
        candidates = [candidate.lower() for candidate in candidates]
        scores = self._compute_context_similarities(candidates, context_words)
        found = [candidate in self._model.words for candidate in candidates]
        problem = [False] * len(scores)
        return ScoringResult(scores, scores, found, problem)

    def _get_context(self, context_before_word: List[str], context_after_word: List[str]) -> List[str]:
        context_before_word = context_before_word[-min(len(context_before_word), self._max_context_words):]
        context_after_word = context_after_word[:min(len(context_after_word), self._max_context_words)]
        context = []
        for word in context_before_word + context_after_word:
            if word not in self._model.words and word.lower() != word:
                word = word.lower()
            if word in self._model.words:
                context.append(word)
        return context

    def _compute_context_similarities(self, candidates: List[str], context_words: List[str]) -> List[float]:
        in_model_candidates = [candidate for candidate in candidates if candidate in self._model.words]
        in_model_similarities = self._model.compute_words_similarity_to_context(in_model_candidates, context_words)
        similarity_by_candidate = {candidate: similarity for candidate, similarity in zip(in_model_candidates, in_model_similarities)}
        similarities = [similarity_by_candidate.get(candidate, 0) for candidate in candidates]
        return similarities
