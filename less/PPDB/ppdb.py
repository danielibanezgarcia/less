import nltk

from .part_of_speech import PartOfSpeech
from .word_dictionary import WordDictionary


class SpanishPpdb(WordDictionary):
    # http://paraphrase.org/#/download
    def __init__(self, dict_path):
        super().__init__(cache=True)

        self.dict_path = dict_path

        self.score_threshold = self.get_default_score_thresholds() # TODO: support other filtering
        self.is_synonym = True  # TODO: antonyms

        self._init()

    def _init(self):
        self.dict = {}
        self.read(self.dict_path)

    @classmethod
    def get_default_score_thresholds(cls):
        return 1.9

    def read(self, model_path):
        with open(model_path, 'r', encoding='utf-8') as f:
            for line in f:

                if '\\ x' in line or 'xc3' in line:
                    continue

                fields = line.split(' ||| ')
                score = float(fields[0].strip())
                if score < self.get_default_score_thresholds():
                    continue
                constituents = fields[1].strip()[1:-1].split('/')
                phrase = fields[2].strip()
                paraphrase = fields[3].strip()

                # filter multiple words
                if len(phrase.split()) != len(paraphrase.split()):
                    continue

                if phrase not in self.dict:
                    self.dict[phrase] = {}

                part_of_speeches = [pos for con in constituents for pos in PartOfSpeech.constituent2pos(con)]

                for pos in part_of_speeches:
                    if pos not in self.dict[phrase]:
                        self.dict[phrase][pos] = []

                    self.dict[phrase][pos].append({
                        'phrase': phrase,
                        'part_of_speech': pos,
                        'synonym': paraphrase,
                        'score': score
                    })

    def predict(self, word, pos=None):
        if pos is None:
            candidates = []
            if word not in self.dict:
                return candidates

            for pos in self.dict[word]:
                for candidate in self.dict[word][pos]:
                    candidates.append(candidate['synonym'])

            return candidates

        if word in self.dict and pos in self.dict[word]:
            return [candidate['synonym'] for candidate in self.dict[word][pos]]

        return []

    def pos_tag(self, tokens):
        return nltk.pos_tag(tokens)
