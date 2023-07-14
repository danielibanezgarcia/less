import subprocess as sp
from datetime import datetime, timedelta
from os import path, popen

import jpype.imports

from less.common.util import read_dict_from_yaml_file

jpype.startJVM(
    jvmpath="/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so",
    classpath=['morphological-generator-jar-with-dependencies.jar']
)
from edu.upf.taln.morphogen.generator import Generator

from less.document.build import DocumentBuilder
from less.embedding.default import DefaultEmbeddingModel
from less.experiment.common.find import WordCandidateFinder
from less.experiment.common.rank import CandidateRanker
from less.experiment.horn.simplify import HornSimplifier
from less.freeling.analyze import FreelingAnalyzer
from less.inflection import SpanishInflectionResolver
from less.scoring.factory import ScorersFactory
from less.stemming import SpanishStemmer


class EfficiencyStats:
    def __init__(
        self,
        disk_usage_in_mb: float,
        cpu_usage_in_mb: float,
        gpu_usage_in_mb: float,
        load_time_in_secs: int,
        processing_time_in_secs: int,
    ):
        self._disk_usage_in_mb = disk_usage_in_mb
        self._cpu_usage_in_mb = cpu_usage_in_mb
        self._gpu_usage_in_mb = gpu_usage_in_mb
        self._load_time_in_secs = load_time_in_secs
        self._processing_time_in_secs = processing_time_in_secs

    @property
    def disk_usage_in_mb(self) -> float:
        return self._disk_usage_in_mb

    @property
    def cpu_usage_in_mb(self) -> float:
        return self._cpu_usage_in_mb

    @property
    def gpu_usage_in_mb(self) -> float:
        return self._gpu_usage_in_mb

    @property
    def load_time_in_secs(self) -> int:
        return self._load_time_in_secs

    @property
    def processing_time_in_secs(self) -> int:
        return self._processing_time_in_secs


def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  return memory_free_values[0]


def get_cpu_memory() -> int:
    total_memory, used_memory, free_memory = map(int, popen('free -t -m').readlines()[-1].split()[1:])
    return used_memory


def compute_efficiency(file_path: str) -> EfficiencyStats:
    file_size = path.getsize(file_path)
    disk_usage = file_size / 1024 / 1024
    file_size = path.getsize("data/ngrams/es-2-grams.pkl")
    disk_usage += file_size / 1024 / 1024
    start_time = datetime.now()
    start_gpu_memory = get_gpu_memory()
    start_cpu_memory = get_cpu_memory()
    horn_simplifier = build_simplifier()
    end_cpu_memory = get_cpu_memory()
    end_gpu_memory = get_gpu_memory()
    end_time = datetime.now()
    cpu_usage = end_cpu_memory - start_cpu_memory
    gpu_usage = start_gpu_memory - end_gpu_memory
    load_processing_time = end_time - start_time
    start_time = datetime.now()
    horn_simplifier.simplify("data/horn/lex.mturk.es_10.txt", "data/horn/performance_10_candidates.txt")
    end_time = datetime.now()
    elapsed_processing_time = end_time - start_time
    return EfficiencyStats(disk_usage, cpu_usage, gpu_usage, load_processing_time.seconds, elapsed_processing_time.seconds)


def build_simplifier() -> HornSimplifier:
    config = read_dict_from_yaml_file('config.yml')

    lang = config["lang"]
    max_initial_candidates = config["max_initial_candidates"]
    max_context_window = config["max_context_window"]
    ngrams_dir = config["ngrams_dir"]
    embedding_model = DefaultEmbeddingModel(config["embeddings_config"]["model_path"])
    scorer_names = config["scorers"]
    freeling_data_dir = config["freeling_data_dir"]

    freeling_analyzer = FreelingAnalyzer(lang, freeling_data_dir)
    morph_generator = Generator(config["morph_generator_config_path"], False)

    stemmer = SpanishStemmer()
    inflection_resolver = SpanishInflectionResolver(morph_generator)
    scorers = ScorersFactory(lang, embedding_model, max_context_window, ngrams_dir).create(scorer_names)

    document_builder = DocumentBuilder(lang, freeling_analyzer)
    candidate_finder = WordCandidateFinder(embedding_model, document_builder, stemmer, inflection_resolver, max_initial_candidates)
    candidate_ranker = CandidateRanker(scorers)
    horn_simplifier = HornSimplifier(document_builder, candidate_finder, candidate_ranker, max_initial_candidates)

    return horn_simplifier


def compute_reduction(metric: float, original_metric: float) -> float:
    difference = original_metric - metric
    reduction_ratio = 0.0 if original_metric == 0 else (difference / original_metric)
    return reduction_ratio


def print_stats(name: str, pstats: EfficiencyStats, original_pstats: EfficiencyStats):
    disk_reduction = compute_reduction(pstats.disk_usage_in_mb, original_pstats.disk_usage_in_mb)
    cpu_reduction = compute_reduction(pstats.cpu_usage_in_mb, original_pstats.cpu_usage_in_mb)
    gpu_reduction = compute_reduction(pstats.gpu_usage_in_mb, original_pstats.gpu_usage_in_mb)
    load_reduction = compute_reduction(pstats.load_time_in_secs, original_pstats.load_time_in_secs)
    performance_reduction = compute_reduction(pstats.processing_time_in_secs, original_pstats.processing_time_in_secs)
    with open(f"data/efficiency/{name}.txt", "w") as file:
        file.write(f"{name} embeddings\n")
        file.write(f" - Disk Size:    {pstats.disk_usage_in_mb:.2f} MB ({disk_reduction:.2%} reduction)\n")
        file.write(f" - CPU Size:     {pstats.cpu_usage_in_mb:.2f} MB ({cpu_reduction:.2%} reduction)\n")
        file.write(f" - GPU Size:     {pstats.gpu_usage_in_mb:.2f} MB ({gpu_reduction:.2%} reduction)\n")
        file.write(
            f" - Load Time:    {str(timedelta(seconds=pstats.load_time_in_secs))} secs. ({load_reduction:.2%} reduction)\n"
        )
        file.write(
            f" - Process Time: {str(timedelta(seconds=pstats.processing_time_in_secs))} secs. ({performance_reduction:.2%} reduction)\n"
        )


if __name__ == "__main__":
    original_stats = compute_efficiency("data/embeddings/cc.es.300.vec")
    print_stats("lightls_300", original_stats, original_stats)
