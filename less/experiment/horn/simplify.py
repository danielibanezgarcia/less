from typing import List, Tuple

from tqdm import tqdm

from less.common.util import write_lines_into_txt_file
from less.document import Token
from less.document.build import DocumentBuilder
from less.experiment.common.find import WordCandidateFinder
from less.experiment.common.rank import CandidateRanker
from less.experiment.horn.common import LexMturkFile


class HornSimplifier:

    def __init__(
        self,
        document_builder: DocumentBuilder,
        word_candidate_finder: WordCandidateFinder,
        ranker: CandidateRanker,
        max_initial_candidates: int,
    ):
        self._document_builder = document_builder
        self._candidate_finder = word_candidate_finder
        self._ranker = ranker
        self._max_initial_candidates = max_initial_candidates

    def simplify(self, input_file_path: str, candidates_file_path: str):
        lexmturk_file = LexMturkFile.load(input_file_path)
        candidates_list = []
        pbar = tqdm(unit="target-word", ncols=80, total=len(lexmturk_file), unit_scale=True, unit_divisor=1,)
        for line in lexmturk_file.lines:
            print(line)
            pbar.update(1)
            candidates = self._get_simpler_words(line.sentence, line.word_start)
            candidates_list.append(candidates)
        self._persist_to_file(candidates_list, candidates_file_path)

    def _get_simpler_words(self, sentence: str, word_start: int) -> List[str]:
        target_word_token, context_before_word, context_after_word = self._extract_sentence_parts(sentence, word_start)
        raw_candidates = [target_word_token.text] + self._candidate_finder.find(target_word_token, context_before_word, context_after_word)
        ranked_candidates = self._ranker.rank(raw_candidates, target_word_token, context_before_word, context_after_word)
        cased_candidates = [candidate.uppercase() for candidate in ranked_candidates] if target_word_token.capitalized else ranked_candidates
        return [candidate.word for candidate in cased_candidates]

    def _extract_sentence_parts(self, sentence: str, word_start: int) -> Tuple[Token, List[str], List[str]]:
        doc = self._document_builder(sentence)
        target_word_token = doc.find_token_at(word_start)
        context_before_word = [token.text for token in doc.find_sentence_at(word_start).get_context_before(target_word_token)]
        context_after_word = [token.text for token in doc.find_sentence_at(word_start).get_context_after(target_word_token)]
        return target_word_token, context_before_word, context_after_word

    def _persist_to_file(self, candidates_list: List[List[str]], results_file_path: str):
        candidates_lines = ['\t'.join(candidates) for candidates in candidates_list]
        write_lines_into_txt_file(candidates_lines, results_file_path)
