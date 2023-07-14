from typing import List

from less.document import Document, EntityMention, NumericExpression, Token


def get_tokens_between_indices(tokens: List[Token], start_index: int, end_index: int) -> List[Token]:
    return [token for token in tokens if token.start_index >= start_index and token.end_index <= end_index]


def get_entities_between_indices(entities: List[EntityMention], start_index: int, end_index: int) -> List[EntityMention]:
    return [entity for entity in entities if entity.start_index >= start_index and entity.end_index <= end_index]


def get_numeric_expressions_between_indices(num_exps: List[NumericExpression], start_index: int, end_index: int) -> List[NumericExpression]:
    return [num_exp for num_exp in num_exps if num_exp.start_index >= start_index and num_exp.end_index <= end_index]


def find_fixable(doc: Document, fixable_pos: List[str]) -> List[Token]:
    return [token for token in doc.words if token.pos in fixable_pos]
