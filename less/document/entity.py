from typing import List

from spacy import explain as spacy_explain

from less.document import EntityMention, Token
from less.document.utils import get_tokens_between_indices


class SpacyEntityExtractor:

    def __init__(self, spacy_numeric_labels: List[str]):
        self._spacy_numeric_labels = set(spacy_numeric_labels)

    def extract(self, spacy_doc, tokens: List[Token]) -> List[EntityMention]:
        entity_mentions = []
        for ent in spacy_doc.ents:
            if ent.label_ not in self._spacy_numeric_labels:
                entity_tokens = get_tokens_between_indices(tokens, ent.start_char, ent.end_char)
                entity_mention = EntityMention(ent.text, ent.start_char, entity_tokens, explanation=spacy_explain(ent.label_))
                entity_mentions.append(entity_mention)
        return entity_mentions
