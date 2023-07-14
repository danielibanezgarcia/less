from abc import ABC
from typing import List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from less.embedding.common import EmbeddingModel


class DefaultEmbeddingModel(EmbeddingModel, ABC):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._vectors = np.zeros((self._rows, self._cols), dtype=np.float32)
        with open(model_path, 'r', encoding='utf-8') as f:
            if self._has_header:
                next(f)
            i = 0
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)
                self._vectors[i] = vector
                self._word2idx[word] = i
                self._words.append(word)
                i += 1

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def word_to_vector(self, word: str) -> Optional[np.ndarray]:
        index = self._word2idx.get(word)
        if index is None:
            return None
        return self._vectors[index]

    def index_to_vector(self, word_index: int) -> np.ndarray:
        return self._vectors[word_index]

    def indices_to_array(self, words_indices: List[int]) -> np.ndarray:
        return self._vectors[words_indices, :]

    def words_to_array(self, words: List[str]) -> np.ndarray:
        words_array = np.zeros((len(words), self._cols))
        for i, word in enumerate(words):
            words_array[i] = self.word_to_vector(word)
        return words_array

    def compute_words_distance(self, word1: str, word2: str) -> float:
        word_vector1 = self.word_to_vector(word1)
        word_vector2 = self.word_to_vector(word2)
        if word_vector1 is None or word_vector2 is None:
            return 1.0
        return self.compute_vector_to_vector_distance(word_vector1, word_vector2)

    def compute_vector_to_vector_distance(self, word_vector1: np.ndarray, word_vector2: np.ndarray) -> float:
        return self.compute_array_to_array_distances(word_vector1.reshape(1, len(word_vector1)), word_vector2.reshape(1, len(word_vector2)))[0][0]

    def compute_vector_to_array_distances(self, word_vector: np.ndarray, word_array: np.ndarray) -> np.ndarray:
        return self.compute_array_to_array_distances(word_vector.reshape(1, len(word_vector)), word_array)[0]

    def compute_array_to_array_distances(self, word_array1: np.ndarray, word_array2: np.ndarray) -> np.ndarray:
        return cdist(word_array1, word_array2, "cosine")

    def compute_words_similarity_to_context(self, words: List[str], context: List[str]) -> List[float]:
        words_array = self.words_to_array(words)
        context_array = self.words_to_array(context)
        distances = self.compute_array_to_array_distances(words_array, context_array)
        similarities = 1.0 - np.clip(distances, 0.0, 1.0)
        similarities_mean = np.mean(similarities, axis=1)
        return similarities_mean.tolist()

    def find_closest_words_from_word(self, word: str, n: int) -> List[str]:
        word_vector = self.word_to_vector(word)
        return self.find_closest_words_from_vector(word_vector, n)

    def find_closest_indices_from_index(self, word_index: int, n: int) -> List[int]:
        word_vector = self.index_to_vector(word_index)
        return self.find_closest_indices_from_vector(word_vector, n)

    def find_closest_words_from_index(self, word_index: int, n: int) -> List[str]:
        word_indices = self.find_closest_indices_from_index(word_index, n)
        return [self._words[i] for i in word_indices]

    def find_closest_list_of_indices_from_indices(self, words_indices: List[int], n: int) -> List[List[int]]:
        words_array = self.indices_to_array(words_indices)
        return self.find_closest_indices_list_from_array(words_array, n)

    def find_closest_words_from_vector(self, word_vector: np.ndarray, n: int) -> List[str]:
        if word_vector is None:
            return []
        closest_indices = self.find_closest_indices_from_vector(word_vector, n)
        return [self._words[i] for i in closest_indices]

    def find_closest_indices_from_vector(self, word_vector: np.ndarray, n: int) -> List[int]:
        if word_vector is None:
            return []
        distances = cdist(word_vector.reshape(1, self._cols), self._vectors, "cosine")[0]
        closest_indices = np.argpartition(distances, range(n + 1))[1: n + 1]
        return list(closest_indices)

    def find_closest_indices_list_from_array(self, words_array: np.ndarray, n: int) -> List[List[int]]:
        distances = cdist(words_array, self._vectors, "cosine")
        closest_indices = np.argpartition(distances, range(n + 1))[:, 1: n + 1]
        return [list(indices) for indices in closest_indices]

    def find_closest_words_list_from_indices(self, words_indices: List[int], n: int) -> List[List[str]]:
        return [self.find_closest_words_from_index(i, n) for i in words_indices]
