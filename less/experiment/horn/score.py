from typing import List, Tuple

from less.document.build import DocumentBuilder
from less.experiment.horn.common import CandidatesFile, LexMturkFile


class HornScorer:

    def __init__(self, doc_builder: DocumentBuilder):
        self._doc_builder = doc_builder

    def score(self, lex_mturk_file_path: str, candidates_file_path: str, max_candidates: int) -> Tuple[float, float, float]:
        lex_mturk_file = LexMturkFile.load(lex_mturk_file_path)
        candidates_file = CandidatesFile.load(candidates_file_path)
        if len(lex_mturk_file) != len(candidates_file):
            raise ValueError(f"The length of annotations file ({len(lex_mturk_file)}) should match the length of candidates file ({len(candidates_file)})")
        target_words, annotations_list, candidates_list = self._lemmatize_lists(
            lex_mturk_file.target_words, lex_mturk_file.annotations_list, candidates_file.candidates_list, max_candidates
        )
        return self._compute_metrics(target_words, annotations_list, candidates_list)

    def _lemmatize_lists(self, target_words: list, annotations_list: list, candidates_list: list, max_candidates: int) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        lemmatized_target_words = self._clean_words(target_words)
        lemmatized_annotations_list = [self._clean_words(annotations) for annotations in annotations_list]
        lemmatized_candidates_list = [self._clean_words(candidates)[:max_candidates] for candidates in candidates_list]
        return lemmatized_target_words, lemmatized_annotations_list, lemmatized_candidates_list

    def _clean_words(self, words: List[str]) -> List[str]:
        return [word.lower() for word in words]

    def _compute_metrics(self, target_words: List[str], annotations_list: List[List[str]], candidates_list: List[List[str]]) -> Tuple[float, float, float]:
        matching_candidates_found_n = 0
        candidates_changed_n = 0
        matching_candidates_found_and_changed_n = 0
        total_n = 0

        for target_word, annotations, candidates in zip(target_words, annotations_list, candidates_list):
            total_n += 1
            matching_candidates = list(set(candidates).intersection(set(annotations)))
            if len(matching_candidates) > 0:
                matching_candidates_found_n += 1
            if self._contains_changes(candidates, target_word):
                candidates_changed_n += 1
            if len(matching_candidates) > 0 and self._contains_changes(matching_candidates, target_word):
                matching_candidates_found_and_changed_n += 1
            elif len(matching_candidates) > 0:
                print()

        precision = matching_candidates_found_n / total_n
        accuracy = matching_candidates_found_and_changed_n / total_n
        changed = candidates_changed_n / total_n
        return precision, accuracy, changed

    def _contains_changes(self, words: List[str], target_word: str) -> bool:
        changes = [word for word in words if word != target_word]
        return len(changes) > 0

    # def _compute_metrics2(self, target_words: List[str], annotations_list: List[List[str]], candidates_list: List[List[str]]) -> Tuple[float, float, float]:
    #     system_changes_n = 0
    #     system_changes_in_human_annotations_n = 0
    #     total_n = len(target_words)
    #
    #     for target_word, annotations, candidates in zip(target_words, annotations_list, candidates_list):
    #         system_changes = {candidate for candidate in candidates}
    #         human_annotations = set(annotations)
    #         # human_changes = {annotation for annotation in annotations if annotation != target_word}
    #         system_changes_in_human_annotations = system_changes.intersection(human_annotations)
    #
    #         if len(system_changes) > 0:
    #             system_changes_n += 1
    #         if len(system_changes_in_human_annotations) > 0:
    #             system_changes_in_human_annotations_n += 1
    #
    #     precision = system_changes_in_human_annotations_n / system_changes_n
    #     accuracy = 1  # matching_candidates_found_and_changed_n / total_n
    #     changed = system_changes_n / total_n
    #     return precision, accuracy, changed
