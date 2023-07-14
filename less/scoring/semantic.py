from typing import List

from less.embedding.common import EmbeddingModel
from less.scoring.common import ScoringResult, WordFeatureScorer


class WordSemanticScorer(WordFeatureScorer):
    def __init__(self, model: EmbeddingModel):
        super().__init__("Semantic similarity")
        self._model = model

    def score(self, candidates: List[str], target_word: str, context_before_word: List[str], context_after_word: List[str]) -> ScoringResult:
        semantic_distances = [self._model.compute_words_distance(target_word, candidate) for candidate in candidates]
        scores = [1.0 - semantic_distance for semantic_distance in semantic_distances]
        found = [True] * len(scores)
        problem = [False] * len(scores)
        return ScoringResult(scores, semantic_distances, found, problem)

    def _get_words_distance(self, word1: str, word2: str) -> float:
        embeddings1 = self._model.word_to_vector(word1)
        if embeddings1 is None and word1.lower() != word1:
            embeddings1 = self._model.word_to_vector(word1)
        if embeddings1 is None:
            return 1.0
        embeddings2 = self._model.word_to_vector(word2)
        if embeddings2 is None and word2.lower() != word2:
            embeddings2 = self._model.word_to_vector(word2)
        if embeddings2 is None:
            return 1.0
        return self._model.compute_vector_to_vector_distance(embeddings1, embeddings2)
