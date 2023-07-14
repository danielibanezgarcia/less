from typing import List

from less.embedding.common import EmbeddingModel
from less.scoring.common import WordFeatureScorer
from less.scoring.context import ContextSimilarityScorer
from less.scoring.fluency import WordFluencyScorer
from less.scoring.frequency import FrequencyScorer
from less.scoring.semantic import WordSemanticScorer


class ScorersFactory:

    def __init__(self, lang, embedding_model: EmbeddingModel, max_context_window: int, ngrams_dir: str):
        self._lang = lang
        self._embedding_model = embedding_model
        self._max_context_window = max_context_window
        self._ngrams_dir = ngrams_dir

    def create(self, scorer_names: List[str]) -> List[WordFeatureScorer]:
        scorer_names_set = set(scorer_names)
        scorers = []
        if "cosine_similarity" in scorer_names_set:
            scorers.append(WordSemanticScorer(self._embedding_model)),
        if "context_similarity" in scorer_names_set:
            scorers.append(ContextSimilarityScorer(self._embedding_model, self._max_context_window)),
        if "fluency" in scorer_names_set:
            scorers.append(WordFluencyScorer(self._lang, self._ngrams_dir))
        if "frequency" in scorer_names_set:
            scorers.append(FrequencyScorer(self._lang))
        return scorers
