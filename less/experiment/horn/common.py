from typing import List

from less.common.util import read_lines_from_csv_file, read_lines_from_txt_file


class LexMturkLine:

    def __init__(self, sentence: str, target_word: str, annotations: List[str]):
        self._sentence = sentence
        self._target_word = target_word
        self._annotations = list(annotations)
        self._annotations_count = {}
        for annotation in annotations:
            if annotation not in self._annotations_count:
                self._annotations_count[annotation] = 0
            self._annotations_count[annotation] += 1

    @property
    def sentence(self) -> str:
        return self._sentence

    @property
    def target_word(self) -> str:
        return self._target_word

    @property
    def word_start(self) -> int:
        return self._sentence.lower().find(self._target_word.lower())

    @property
    def annotations(self) -> List[str]:
        return [annotation.strip() for annotation in self._annotations if len(annotation.strip())]

    def annotation_count(self, annotation: str) -> int:
        return self._annotations_count.get(annotation.strip().lower(), 0)

    def __str__(self) -> str:
        return f'{self._sentence} {self._target_word} {self._annotations}'


class LexMturkFile:

    def __init__(self, file_path: str, lines: List[LexMturkLine]):
        self._file_path = file_path
        self._lines = list(lines)

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def target_words(self) -> List[str]:
        return [line.target_word for line in self._lines]

    @property
    def annotations_list(self) -> List[List[str]]:
        return [line.annotations for line in self._lines]

    @property
    def lines(self) -> List[LexMturkLine]:
        return list(self._lines)

    @staticmethod
    def load(file_path: str):
        lines = [LexMturkFile._parse_line(line) for line in list(read_lines_from_csv_file(file_path, '\t'))]
        return LexMturkFile(file_path, lines)

    @staticmethod
    def _parse_line(line: List[str]) -> LexMturkLine:
        return LexMturkLine(line[0], line[1], line[2:])

    def __len__(self):
        return len(self._lines)

    def __iter__(self):
        for line in self._lines:
            yield line


class CandidatesFile:

    def __init__(self, file_path: str, candidates_list: List[List[str]]):
        self._file_path = file_path
        self._candidates_list = list(candidates_list)

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def candidates_list(self) -> List[List[str]]:
        return list(self._candidates_list)

    @staticmethod
    def load(file_path: str):
        candidates_list = [candidates_line.split('\t') for candidates_line in read_lines_from_txt_file(file_path)]
        return CandidatesFile(file_path, candidates_list)

    def __len__(self):
        return len(self._candidates_list)

    def __iter__(self):
        for candidates in self._candidates_list:
            yield candidates


