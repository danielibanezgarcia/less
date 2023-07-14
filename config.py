import jpype.imports

try:
    jpype.startJVM(
        jvmpath="/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so",
        classpath=['morphological-generator-jar-with-dependencies.jar']
    )
except:
    pass
from edu.upf.taln.morphogen.generator import Generator

from less.embedding.default import DefaultEmbeddingModel
from less.experiment.common.find import WordCandidateFinder
from less.experiment.common.rank import CandidateRanker
from less.experiment.horn.simplify import HornSimplifier
from less.inflection import SpanishInflectionResolver
from less.scoring.factory import ScorersFactory
from less.spacy.nlp import SpacyNlpFactory
from less.stemming import SpanishStemmer


from less.common.util import read_dict_from_yaml_file
from less.document.build import DocumentBuilder
from less.experiment.horn.score import HornScorer
from less.freeling.analyze import FreelingAnalyzer

config = read_dict_from_yaml_file('config.yml')
lang = config["lang"]
max_initial_candidates = config["max_initial_candidates"]
max_context_window = config["max_context_window"]
ngrams_dir = config["ngrams_dir"]
scorer_names = config["scorers"]
freeling_data_dir = config["freeling_data_dir"]
horn_lex_mturk_file = config["horn"]["lex_mturk_file"]
horn_scores_file = config["horn"]["scores_file"]
horn_candidates_file = config["horn"]["candidates_file"]
morph_generator = Generator(config["morph_generator_config_path"], False)
embedding_model = DefaultEmbeddingModel(config["embeddings_config"]["model_path"])
spacy_corpus_by_lang = config["spacy"]["corpora"]

spacy_nlp = SpacyNlpFactory(spacy_corpus_by_lang).create(lang)
freeling_analyzer = FreelingAnalyzer(lang, freeling_data_dir)
stemmer = SpanishStemmer()
inflection_resolver = SpanishInflectionResolver(morph_generator)
scorers = ScorersFactory(lang, ngrams_dir, embedding_model, max_context_window).create(scorer_names)

document_builder = DocumentBuilder(lang, freeling_analyzer)
candidate_finder = WordCandidateFinder(embedding_model, document_builder, stemmer, inflection_resolver, max_initial_candidates)
candidate_ranker = CandidateRanker(scorers)

horn_simplifier = HornSimplifier(document_builder, candidate_finder, candidate_ranker)
horn_scorer = HornScorer(document_builder)
