from typing import List

from less.document import Token
from less.scoring.common import WordFeatureScorer

INVALID_SCORE_RANKING_VALUE = 10000

def rank_values(scores: List[float], problem: List[bool] = None, reverse: bool = True) -> List[int]:
    problem = problem or [False] * len(scores)
    indexes_scores_problem = zip(range(len(scores)), scores, problem)
    sorted_indexes_scores_problem = sorted(indexes_scores_problem, key=lambda x: x[1], reverse=reverse)
    i = 0
    rankings = [-1] * len(scores)
    last_score = -1
    for index, score, problem in sorted_indexes_scores_problem:
        if problem:
            rankings[index] = INVALID_SCORE_RANKING_VALUE
            continue
        if score != last_score:
            i += 1
        rankings[index] = i
        last_score = score
    return rankings


def merge_rankings(rankings: List[List[float]]) -> List[int]:
    rankings_sum_per_candidate = [sum(ranking) for ranking in rankings]
    return rank_values(rankings_sum_per_candidate, reverse=False)


class RankedWordCandidate:
    def __init__(self, word: str, merged_rank: float):
        self._word = word
        self._merged_rank = merged_rank

    @property
    def word(self) -> str:
        return self._word

    @property
    def merged_rank(self) -> float:
        return self._merged_rank

    def uppercase(self):
        self._word = self._word[0].upper() + self._word[1:]
        return self

    def __str__(self) -> str:
        return f"{self._word} - {self.merged_rank}"


class CandidateRanker:

    def __init__(
        self,
        scorers: List[WordFeatureScorer],
    ):
        self._scorers = list(scorers or [])
        self._scorers_weights = [1.0] * len(scorers)

    def rank(self, candidates: List[str], target: Token, context_before_word: List[str], context_after_word: List[str]) -> List[RankedWordCandidate]:
        all_rankings = []
        for scorer in self._scorers:
            score_result = scorer.score(candidates, target.text, context_before_word, context_after_word)
            ranking = rank_values(score_result.scores, score_result.problem)
            all_rankings.append(ranking)
        rankings_per_candidate = [
            [
                all_rankings[j][i]*weight
                for j, weight in zip(range(len(self._scorers)), self._scorers_weights)
            ]
            for i in range(len(candidates))
        ]
        merged_ranking = merge_rankings(rankings_per_candidate)
        ranked_candidates = [RankedWordCandidate(candidate, rank) for candidate, rank in zip(candidates, merged_ranking)]
        sorted_ranked_candidates = sorted(ranked_candidates, key=lambda candidate: candidate.merged_rank)
        return sorted_ranked_candidates
