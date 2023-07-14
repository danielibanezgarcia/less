from os import path

from less.freeling import pyfreeling as freeling


class FreelingAnalyzer:

    def __init__(self, lang: str, data_dir: str):

        ## Modify this line to be your FreeLing installation directory
        self._lang = lang
        freeling.util_init_locale("default")

        # create language analyzer
        self._language_analyzer = freeling.lang_ident(path.join(data_dir, "common", "lang_ident", "ident.dat"))

        # create options set for maco analyzer. Default values are Ok, except for data files.
        options = freeling.maco_options(lang)
        options.set_data_files(
            "",
            path.join(data_dir, "common", "punct.dat"),
            path.join(data_dir, lang, "dicc.src"),
            path.join(data_dir, lang, "afixos.dat"),
            "",
            path.join(data_dir, lang, "locucions.dat"),
            path.join(data_dir, lang, "np.dat"),
            path.join(data_dir, lang, "quantities.dat"),
            path.join(data_dir, lang, "probabilitats.dat"),
        )

        # create analyzers
        self._tokenizer = freeling.tokenizer(path.join(data_dir, lang, "tokenizer.dat"))
        self._splitter = freeling.splitter(path.join(data_dir, lang, "splitter.dat"))
        self._morpho = freeling.maco(options)

        # activate mmorpho modules to be used in next call
        self._morpho.set_active_options(
            False, True, True, True,   # select which among created
            True, True, False, False,  # submodules are to be used.
            True, True, True, True,    # default: all created submodules are used
        )

        #PARAMETERS
        #set_active_options(
        # bool umap --> USER MAP                   (False)
        # bool num --> number detection
        # bool pun  --> punctuation detection
        # bool dat -->  date/time detection
        # bool dic --> dictionary search
        # bool aff  --> affixes
        # bool comp --> compound analysis           (False)
        # bool rtk --> retokenize contractions       (False)
        # bool mw  --> multiwords detection
        # bool ner --> named entity recognition
        # bool qt --> quantities, ratios, and percentages detection
        # bool prb --> probabilities

        # create tagger, sense anotator, and parsers
        self._tagger = freeling.hmm_tagger(path.join(data_dir, lang, "tagger.dat"), False, 1)

    def process(self, text) -> list:
        #ls = sen.analyze(ls)
        session_id = self._splitter.open_session()
        tokens = self._tokenizer.tokenize(text)
        ls = self._splitter.split(session_id, tokens, True)
        ls = self._morpho.analyze(ls)
        ls = self._tagger.analyze(ls)
        self._splitter.close_session(session_id)
        return ls
