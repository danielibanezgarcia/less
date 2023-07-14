from abc import ABC, abstractmethod
from typing import List, Optional

from edu.upf.taln.morphogen.generator import Generator
from lemminflect import getInflection

from less.document import Token


class InflectionResolver(ABC):
    @abstractmethod
    def resolve(
        self, candidate: Token, original: Token, context_before_word: List[str], context_after_word: List[str]
    ) -> Optional[str]:
        raise NotImplementedError


class SpanishInflectionResolver(InflectionResolver):

    def __init__(self, morph_generator: Generator):
        self._morph_generator = morph_generator

    def resolve(
        self, candidate: Token, original: Token, context_before_word: List[str], context_after_word: List[str]
    ) -> Optional[str]:
        forms = self._morph_generator.getMorphologicalGenerationForms_Freeling_PoS(candidate.lemma, original.tag)
        if forms is None:
            return None
        forms = forms.toArray()
        return forms[0] if forms.length > 0 else None


class EnglishInflectionResolver(InflectionResolver):
    def resolve(
        self, candidate: Token, original: Token, context_before_word: List[str], context_after_word: List[str]
    ) -> Optional[str]:
        inflected_token = getInflection(candidate.lemma, original.tag)
        if len(inflected_token) == 0:
            return None
        inflected_candidate = inflected_token[0]
        if inflected_candidate is None:
            return None
        if len(inflected_candidate) == 0:
            return None
        if inflected_candidate.startswith("-"):
            return None
        return inflected_candidate
