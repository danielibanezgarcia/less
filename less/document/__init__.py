from abc import ABC
from enum import Enum
from typing import List, Optional

from less.common.util import create_uuid


class TextObject(ABC):

    def __init__(self, text: str, start_index: int):
        self._id = create_uuid()
        self._text = text
        self._length_in_chars = len(text)
        self._start_index = start_index
        self._end_index = start_index + len(text)

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def capitalized(self) -> bool:
        return self.text[0].isupper()

    @property
    def start_index(self) -> int:
        return self._start_index

    @property
    def end_index(self) -> int:
        return self._end_index

    @property
    def length_in_chars(self) -> int:
        return self._length_in_chars

    def is_included_in(self, text_object) -> bool:
        return text_object.start_index <= self.start_index <= self.end_index <= text_object.end_index

    def overlaps_with(self, text_object) -> bool:
        return (text_object.start_index <= self.end_index <= text_object.end_index) or (self.start_index <= text_object.end_index <= self.end_index)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
        }

    def __str__(self) -> str:
        return self._text

    def __eq__(self, other) -> bool:
        return self.text == other.text and self.start_index == other.start_index

    def __hash__(self):
        return hash(self.text)

    def __lt__(self, other) -> bool:
        return self.start_index < other.start_index

    def __le__(self, other) -> bool:
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other) -> bool:
        return self.end_index > other.end_index

    def __ge__(self, other) -> bool:
        return self.__eq__(other) or self.__gt__(other)


class Token(TextObject):

    def __init__(self, text: str, start_index: int, lemma: str, alpha: bool, frequency: float, pos: str, tag: str, syllables: list):
        super().__init__(text, start_index)
        self._lemma = lemma
        self._syllables = list(syllables)
        self._length_in_syllables = len(syllables)
        self._alpha = alpha
        self._frequency = frequency
        self._pos = pos
        self._tag = tag
        self._number_of_syllables = len(self._syllables)
        self._number_of_chars = len(self._text)

    @property
    def length_in_syllables(self) -> int:
        return self._length_in_syllables

    @property
    def lemma(self) -> str:
        return self._lemma

    @property
    def alpha(self) -> bool:
        return self._alpha

    @property
    def number_of_syllables(self) -> int:
        return self._number_of_syllables

    @property
    def number_of_chars(self) -> int:
        return self._number_of_chars

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def pos(self) -> str:
        return self._pos

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def syllables(self) -> List[str]:
        return list(self._syllables)


class TokenSequence(TextObject):

    def __init__(self, text: str, start_index: int, tokens: List[Token]):
        super().__init__(text, start_index)
        self._tokens = list(tokens)

    @property
    def tokens(self) -> List[Token]:
        return list(self._tokens)

    @property
    def words(self) -> List[Token]:
        return [token for token in self.tokens if token.alpha]

    def get_token_matching_text(self, text: str) -> Optional[Token]:
        lower_text = text.lower()
        for token in self._tokens:
            if token.lower == lower_text:
                return token
        return None


class Entity:

    def __init__(self, concept: str):
        self._id = create_uuid()
        self._concept = concept

    @property
    def id(self) -> str:
        return self._id

    @property
    def concept(self) -> str:
        return self._concept

    def __str__(self) -> str:
        return self.concept


class EntityMention(TokenSequence):

    def __init__(self, text: str, start_index: int, tokens: List[Token], explanation: str = None, entity: Entity = None):
        super().__init__(text, start_index, tokens)
        self._explanation = explanation
        self._entity = entity

    @property
    def explanation(self) -> str:
        return self._explanation

    @property
    def entity(self) -> Entity:
        return self._entity


class NumericExpression(TokenSequence):

    def __init__(self, text: str, start_index: int, tokens: List[Token], explanation: str):
        super().__init__(text, start_index, tokens)
        self._explanation = explanation

    @property
    def explanation(self) -> str:
        return self._explanation


class CohesiveMarkerType(Enum):
    ADDITIVE = "additive"
    COMPARATIVE = "comparative"
    GENERALIZING = "generalizing"
    CAUSAL = "causal"
    CONTRASTIVE = "contrastive"
    TEMPORAL = "temporal"
    SEQUENTIAL = "sequential"
    EMPHASIZING = "emphasizing"
    REPEATING = "repeating"
    EXEMPLIFYING = "exemplifying"
    CONCLUDING = "concluding"
    PHRASAL = "phrasal"
    CONCESSIVE = "concessive"
    LOGICAL_SEMANTIC = "logical_semantic"
    OTHER = "other"
    SUMMATIVE = "summative"
    RESULTATIVE = "resultative"
    RESTATEMENT = "restatement"


class CohesiveMarker(TokenSequence):

    def __init__(self, text: str, start_index: int, tokens: List[Token], marker_type: CohesiveMarkerType):
        super().__init__(text, start_index, tokens)
        self._type = marker_type

    @property
    def type(self) -> CohesiveMarkerType:
        return self._type


class NounPhrase(TokenSequence):

    def __init__(self, text: str, start_index: int, tokens: List[Token]):
        super().__init__(text, start_index, tokens)

    def merge_with(self, noun_phrase):
        first, second = self._get_first_and_second(noun_phrase)
        merged_text = first.text + second.text[len(first.text):]
        merged_tokens = first.tokens + [token for token in second.tokens if token not in first.tokens]
        return NounPhrase(merged_text, first.start_index, merged_tokens)

    def _get_first_and_second(self, noun_phrase) -> tuple:
        if self.start_index == noun_phrase.start_index:
            return (self, noun_phrase) if self.end_index < noun_phrase.end_index else (noun_phrase, self)
        return (self, noun_phrase) if self.start_index < noun_phrase.start_index else (noun_phrase, self)


class Sentence(TokenSequence):

    def __init__(
        self,
        text: str,
        start_index: int,
        tokens: List[Token],
        cohesive_markers: List[CohesiveMarker],
        noun_phrases: List[NounPhrase],
        entities: List[EntityMention],
        numeric_expressions: List[NumericExpression]
    ):
        super().__init__(text, start_index, tokens)
        self._cohesive_markers = list(cohesive_markers)
        self._noun_phrases = list(noun_phrases)
        self._entities = list(entities)
        self._numeric_expressions = list(numeric_expressions)

    @property
    def words(self) -> List[Token]:
        return [token for token in self._tokens if token.alpha]

    @property
    def length_in_words(self) -> int:
        return len(self.words)

    @property
    def syllables(self) -> List[str]:
        return [syllable for token in self._tokens for syllable in token.syllables]

    @property
    def cohesive_markers(self) -> List[CohesiveMarker]:
        return list(self._cohesive_markers)

    @property
    def noun_phrases(self) -> List[NounPhrase]:
        return list(self._noun_phrases)

    @property
    def entities(self) -> List[EntityMention]:
        return list(self._entities)

    @property
    def numeric_expressions(self) -> List[NumericExpression]:
        return list(self._numeric_expressions)

    def get_context_before(self, token: Token) -> List[Token]:
        return [context_token for context_token in self.tokens if context_token < token]

    def get_context_after(self, token: Token) -> List[Token]:
        return [context_token for context_token in self._tokens if context_token > token]

    def to_dict(self) -> dict:
        result = super().to_dict()
        result.update({
            "tokens": [token.to_dict() for token in self._tokens]
        })
        return result

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self._tokens[start: stop: step]
        elif isinstance(key, int):
            return self._tokens[key]


class Paragraph(TokenSequence):

    def __init__(self, text: str, start_index: int, sentences: List[Sentence]):
        super().__init__(text, start_index, [token for sentence in sentences for token in sentence.tokens])
        self._sentences = list(sentences)

    @property
    def sentences(self) -> List[Sentence]:
        return list(self._sentences)

    @property
    def tokens(self) -> List[Token]:
        return [token for sentence in self._sentences for token in sentence.tokens]

    @property
    def words(self) -> List[Token]:
        return [word for sentence in self._sentences for word in sentence.words]

    @property
    def syllables(self) -> List[str]:
        return [syllable for sentence in self._sentences for syllable in sentence.syllables]

    @property
    def cohesive_markers(self) -> List[CohesiveMarker]:
        return [marker for sentence in self._sentences for marker in sentence.cohesive_markers]

    @property
    def noun_phrases(self) -> List[NounPhrase]:
        return [noun_phrase for sentence in self._sentences for noun_phrase in sentence.noun_phrases]

    @property
    def entities(self) -> List[EntityMention]:
        return [entity for sentence in self._sentences for entity in sentence.entities]

    @property
    def numeric_expressions(self) -> List[NumericExpression]:
        return [numeric_exp for sentence in self._sentences for numeric_exp in sentence.numeric_expressions]

    def to_dict(self) -> dict:
        result = super().to_dict()
        result.update({
            "sentences": [sentence.to_dict() for sentence in self._sentences]
        })
        return result

    def __iter__(self):
        return iter(self._sentences)

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self._sentences[start: stop: step]
        elif isinstance(key, int):
            return self._sentences[key]


class Document(TokenSequence):

    def __init__(self, text: str, paragraphs: List[Paragraph]):
        super().__init__(text, 0, [token for paragraph in paragraphs for token in paragraph.tokens])
        self._paragraphs = list(paragraphs)

    @property
    def paragraphs(self) -> List[Paragraph]:
        return list(self._paragraphs)

    @property
    def sentences(self) -> List[Sentence]:
        return [sentence for paragraph in self._paragraphs for sentence in paragraph.sentences]

    @property
    def tokens(self) -> List[Token]:
        return [token for paragraph in self._paragraphs for token in paragraph.tokens]

    @property
    def words(self) -> List[Token]:
        return [word for paragraph in self._paragraphs for word in paragraph.words]

    @property
    def syllables(self) -> List[str]:
        return [syllable for paragraph in self._paragraphs for syllable in paragraph.syllables]

    @property
    def cohesive_markers(self) -> List[CohesiveMarker]:
        return [marker for paragraph in self._paragraphs for marker in paragraph.cohesive_markers]

    @property
    def noun_phrases(self) -> List[NounPhrase]:
        return [noun_phrase for paragraph in self._paragraphs for noun_phrase in paragraph.noun_phrases]

    @property
    def entities(self) -> List[EntityMention]:
        return [entity for paragraph in self._paragraphs for entity in paragraph.entities]

    @property
    def numeric_expressions(self) -> List[NumericExpression]:
        return [numeric_exp for paragraph in self._paragraphs for numeric_exp in paragraph.numeric_expressions]

    def find_token_at(self, index: int) -> Optional[Token]:
        for token in self.tokens:
            if token.start_index <= index <= token.end_index:
                return token
        return None

    def find_sentence_at(self, index: int) -> Optional[Sentence]:
        for sentence in self.sentences:
            if sentence.start_index <= index <= sentence.end_index:
                return sentence
        return None

    def to_dict(self) -> dict:
        result = super().to_dict()
        result.update({
            "paragraphs": [paragraph.to_dict() for paragraph in self._paragraphs]
        })
        return result

    def __iter__(self):
        return iter(self._paragraphs)

    def __len__(self):
        return len(self._paragraphs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self._paragraphs[start: stop: step]
        elif isinstance(key, int):
            return self._paragraphs[key]
