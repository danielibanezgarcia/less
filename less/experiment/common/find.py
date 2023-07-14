from typing import List, Optional

from less.document.build import DocumentBuilder
from less.document.utils import Token
from less.embedding.common import EmbeddingModel
from less.inflection import InflectionResolver
from less.stemming import Stemmer


class WordCandidateFinder:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        document_builder: DocumentBuilder,
        stemmer: Stemmer,
        inflection_resolver: InflectionResolver,
        n_candidates: int
    ):
        self._embedding_model = embedding_model
        self._document_builder = document_builder
        self._stemmer = stemmer
        self._inflection_resolver = inflection_resolver
        self._n_candidates = n_candidates

    def find(self, target_word: Token, context_before_word: List[str], context_after_word: List[str]) -> List[str]:
        initial_candidates = self._get_initial_candidates(target_word)
        non_matching_stem_candidates = self._remove_candidates_with_same_stem(initial_candidates)
        inflected_candidates = self._inflect_candidates(non_matching_stem_candidates, target_word, context_before_word, context_after_word)
        candidates_matching_tag = self._remove_candidates_not_matching_tag(inflected_candidates, target_word, context_before_word, context_after_word)
        candidates_without_target_word = [candidate for candidate in candidates_matching_tag if candidate != target_word.lower]
        return candidates_without_target_word

    def _get_initial_candidates(self, target_word: Token) -> List[str]:
        initial_candidates = self._embedding_model.find_closest_words_from_word(target_word.text, self._n_candidates)
        initial_candidates = [candidate for candidate in initial_candidates if self._alpha_ratio(candidate) > 0.95]
        if target_word.text != target_word.lower:
            initial_candidates.extend(self._embedding_model.find_closest_words_from_word(target_word.lower, self._n_candidates))
        if target_word.lemma and target_word.text.lower() != target_word.lemma.lower():
            initial_candidates.extend(self._embedding_model.find_closest_words_from_word(target_word.lemma, self._n_candidates))
        return initial_candidates

    def _alpha_ratio(self, word: str) -> float:
        alpha_chars = sum([1 for char in word if char.isalpha()])
        alpha_ratio = alpha_chars / len(word)
        return alpha_ratio

    def _inflect_candidates(self, candidates: List[str], target: Token, context_before_word: List[str], context_after_word: List[str]) -> List[str]:
        inflected_candidates = []
        for candidate in candidates:
            inflected_candidate = self._set_inflection(candidate, target, context_before_word, context_after_word)
            if inflected_candidate is not None:
                inflected_candidates.append(inflected_candidate)
        return inflected_candidates

    def _set_inflection(
        self, candidate: str, target_word: Token, context_before_word: List[str], context_after_word: List[str]
    ) -> Optional[str]:
        text = " ".join([word for word in context_before_word] + [candidate] + [word for word in context_after_word])
        candidate_token = self._document_builder(text).get_token_matching_text(candidate)
        if candidate_token is None:
            return None
        inflected_candidate = self._inflection_resolver.resolve(
            candidate_token, target_word, context_before_word, context_after_word
        )
        return None if inflected_candidate is None else str(inflected_candidate)

    def _remove_candidates_with_same_stem(self, candidates: List[str]) -> List[str]:
        non_matching_stem_candidates = []
        visited_stems = set()
        for candidate in candidates:
            candidate_stem = self._stemmer.stem(candidate)
            if candidate_stem not in visited_stems:
                visited_stems.add(candidate_stem)
                non_matching_stem_candidates.append(candidate)
        return non_matching_stem_candidates

    def _remove_candidates_not_matching_tag(
        self, candidates: List[str], target_word: Token, context_before_word: List[str], context_after_word: List[str]
    ) -> List[str]:
        matching_candidates = []
        for candidate in candidates:
            if self._get_candidate_tag(candidate) == target_word.tag:
                matching_candidates.append(candidate)
            elif self._get_candidate_tag_in_context(candidate, context_before_word, context_after_word) == target_word.tag:
                matching_candidates.append(candidate)
        return matching_candidates

    def _get_candidate_tag(self, candidate: str) -> str:
        candidate_token = self._document_builder(candidate).find_token_at(0)
        return candidate_token.tag if candidate_token else ''

    def _get_candidate_tag_in_context(self, candidate: str, context_before_word: List[str], context_after_word: List[str]) -> str:
        candidate_token = self._document_builder(' '.join(context_before_word + [candidate] + context_after_word)).get_token_matching_text(candidate)
        return candidate_token.tag if candidate_token else ''
