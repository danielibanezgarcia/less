from typing import List, Tuple

import wordfreq as wf

from less.document import Document, Paragraph, Sentence, Token
from less.document.utils import get_tokens_between_indices
from less.freeling.analyze import FreelingAnalyzer



class DocumentBuilder:

    def __init__(
        self,
        lang: str,
        freeling_analyzer: FreelingAnalyzer,
    ):
        self._lang = lang
        self._freeling_analyzer = freeling_analyzer

    def __call__(self, text: str) -> Document:
        return self.build(text)

    def build(self, text: str) -> Document:
        freeling_sentences = self._freeling_analyzer.process(text)
        freeling_tokens = [word for sentence in freeling_sentences for word in sentence.get_words()]
        tokens = self._extract_tokens(freeling_tokens)
        sentences = self._extract_sentences(text, freeling_sentences, tokens)
        paragraphs = self._extract_paragraphs(text, sentences)
        document = Document(text, paragraphs)
        return document

    def _extract_tokens(self,freeling_tokens: List[str]) -> List[Token]:
        tokens = []
        for freeling_token in freeling_tokens:
            token = Token(
                freeling_token.get_form(),
                freeling_token.get_span_start(),
                freeling_token.get_lemma(),
                freeling_token.get_form().isalpha(),
                self._compute_freq(freeling_token),
                freeling_token.get_tag()[0],
                freeling_token.get_tag(),
                []
            )
            tokens.append(token)
        return tokens

    def _compute_freq(self, freeling_token: str) -> float:
        freq_val = 0.0
        token_text = freeling_token.get_form()
        token_lemma = freeling_token.get_lemma()
        if token_text.isalpha() or token_lemma.isalpha():
            # max between lemma and text, to get by things like:
            # n't, 've, 're, 'll...
            freq_val = max([wf.zipf_frequency(token_text, self._lang), wf.zipf_frequency(token_lemma, self._lang)])
        return freq_val

    def _extract_sentences(self, text: str, freeling_sentences: List[str], tokens: List[Token]) -> List[Sentence]:
        sentences = []
        for freeling_sent in freeling_sentences:
            sent_start = freeling_sent.get_words()[0].get_span_start()
            sent_end = freeling_sent.get_words()[-1].get_span_finish()
            sentence_tokens = get_tokens_between_indices(tokens, sent_start, sent_end)
            sentence = Sentence(
                text[sent_start: sent_end],
                sent_start,
                sentence_tokens,
                [],
                [],
                [],
                []
            )
            sentences.append(sentence)
        return sentences

    def _extract_paragraphs(self, text: str, sentences: List[Sentence]) -> List[Paragraph]:
        paragraphs = []
        paragraph_sentences = []
        for i, sentence in enumerate(sentences):
            paragraph_sentences.append(sentence)
            is_last_sentence = (i == len(sentences) - 1)
            if sentence.text.endswith("\n") or is_last_sentence:
                start_index = paragraph_sentences[0].start_index
                end_index = sentence.end_index
                paragraph = Paragraph(text[start_index: end_index], sentence.start_index, paragraph_sentences)
                paragraphs.append(paragraph)
        return paragraphs
