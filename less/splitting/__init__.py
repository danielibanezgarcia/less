from typing import List

from less.common.util import create_uuid


class SplittedSentencesMerger:
    def merge_splitted_sentences(self, sentences: List[dict], sentence_id: str, splitted_sentences: List[dict]) -> List[dict]:
        sentence_to_split_index, sentence_to_split = [(i, sentence) for i, sentence in enumerate(sentences) if sentence["id"] == sentence_id][0]
        new_sentences = sentences[:sentence_to_split_index]
        offset = sentence_to_split["start_position"]
        for sentence in splitted_sentences:
            sentence["id"] = create_uuid()
            sentence["start_position"] += offset
            sentence["end_position"] += offset
            new_sentences.append(sentence)
        offset = splitted_sentences[-1]["end_position"] - sentence_to_split["end_position"]
        for sentence in sentences[sentence_to_split_index + 1:]:
            sentence["start_position"] += offset
            sentence["end_position"] += offset
            new_sentences.append(sentence)
        return new_sentences
