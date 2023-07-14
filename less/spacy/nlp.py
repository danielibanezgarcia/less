from typing import Dict

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex


class SpacyNlpFactory:

    def __init__(self, spacy_corpus_by_lang: Dict[str, str]):

        self._nlp_by_lang = {}
        for lang, spacy_corpus in spacy_corpus_by_lang.items():
            nlp = spacy.load(spacy_corpus)
            infixes = nlp.Defaults.prefixes + [r"[./]", r"[-]~", r"(.'.)"]
            infix_re = spacy.util.compile_infix_regex(infixes)
            prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
            suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

            nlp.tokenizer = Tokenizer(
                nlp.vocab,
                prefix_search=prefix_re.search,
                suffix_search=suffix_re.search,
                infix_finditer=infix_re.finditer,
                token_match=nlp.tokenizer.token_match,
            )
            self._nlp_by_lang[lang] = nlp

    def create(self, lang: str):
        return self._nlp_by_lang[lang]

