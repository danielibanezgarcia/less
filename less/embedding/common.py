from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


def get_array_shape(model_path: str) -> Tuple[int, int, bool]:
    with open(model_path, "r", encoding="utf-8") as f:
        first_line = next(f)
        first_line_items = first_line.split()
        if len(first_line_items) == 2:
            return int(first_line_items[0]), int(first_line_items[1]), True
        cols = len(first_line.split()) - 1
        rows = 1
        for _ in f:
            rows += 1
    return rows, cols, False


class EmbeddingModel(ABC):

    def __init__(self, model_path: str):
        self._word2idx = {}
        self._words = []
        self._rows, self._cols, self._has_header = get_array_shape(model_path)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def words(self) -> List[str]:
        return list(self._words)

    def index(self, word: str) -> Optional[int]:
        return self._word2idx.get(word)

    def __len__(self):
        return self.rows

    @abstractmethod
    def word_to_vector(self, word: str) -> Optional[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def index_to_vector(self, word_index: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def words_to_array(self, words: tuple) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def indices_to_array(self, words_indices: List[int]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_words_distance(self, word1: str, word2: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_vector_to_vector_distance(self, word_vector1: np.ndarray, word_vector2: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_array_to_array_distances(self, word_array1: np.ndarray, word_array2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_words_similarity_to_context(self, words: List[str], context_words: List[str]) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_words_from_word(self, word_vector: str, n: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_indices_from_index(self, word_index: int, n: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_words_from_index(self, word_index: int, n: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_list_of_indices_from_indices(self, words_indices: List[int], n: int) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_words_from_vector(self, word_vector: np.ndarray, n: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_indices_from_vector(self, word_vector: np.ndarray, n: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_indices_list_from_array(self, words_array: np.ndarray, n: int) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def find_closest_words_list_from_indices(self, words_indices: List[int], n: int) -> List[List[str]]:
        raise NotImplementedError


