import random
from datetime import datetime
from os import path
from typing import List

import numpy as np
import torch
from nltk import PorterStemmer
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity as cosine
from transformers import BertForMaskedLM, BertTokenizer
from wordfreq import get_frequency_dict

from measure_efficiency import EfficiencyStats, get_gpu_memory, print_stats

from config import horn_scorer
from measure_efficiency import get_cpu_memory
from less.PPDB import SpanishPpdb
from less.experiment.common.rank import RankedWordCandidate
from less.experiment.horn.common import LexMturkFile


EMBEDDINGS_FILE_PATH = "data/embeddings/cc.es.300.vec"
PPDB_FILE_PATH = "data/ppdb/ppdb-2.0-es-lexical"
BERT_FILE_PATH = "/home/daniel/.cache/huggingface/transformers/52382cbe7c1587c6b588daa81eaf247c5e2ad073d42b52192a8cd4202e7429b6.a88ccd19b1f271e63b6a901510804e6c0318089355c471334fe8b71b316a30ab"


class Ranker:

    def __init__(self):
        self.fasttext_dico = None
        self.fasttext_emb = None
        self.ppdb = None
        self.ps = PorterStemmer()
        self.word_count = None

    def getWordmap(self, wordVecPath):
        words = []
        We = []
        f = open(wordVecPath, 'r')
        lines = f.readlines()

        for (n, line) in enumerate(lines):
            if (n == 0):
                print(line)
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            We.append(vect)
            words.append(word)

        f.close()
        return (words, We)

    def _getWordCount(self):
        word2count = get_frequency_dict("es")
        words = len(word2count)
        word2count = {key: int(value * words) for key, value in word2count.items()}
        return word2count

    def read_features(self, word_embeddings, ppdb_path):

        start_time = datetime.now()
        print("---read word frequency----")
        self.word_count = self._getWordCount()
        print(datetime.now() - start_time)
        print("-----finished reading word frequency----")
        print("-----reading word embeddings----")
        start_time = datetime.now()
        self.fasttext_dico, self.fasttext_emb = self.getWordmap(word_embeddings)
        print(datetime.now() - start_time)
        print("-----finished reading word embeddings----")
        print("----loading PPDB ...")
        start_time = datetime.now()
        self.ppdb = SpanishPpdb(ppdb_path)
        print(datetime.now() - start_time)
        print("-----finished loading PPDB----")


def candidates_generation(
    sentence: str, word_to_replace: str, max_candidates: int, tokenizer, language_model, ps: PorterStemmer, device
) -> List[str]:
    masked_sentence = sentence.lower().replace(word_to_replace.lower(), "[MASK]")
    inputs = tokenizer(
        sentence + " [SEP] " + masked_sentence,
        return_tensors="pt"
    )
    mask_index = tokenizer.vocab["[MASK]"]
    mask_pos = inputs["input_ids"].numpy()[0].tolist().index(mask_index)

    inputs = inputs.to(device)
    outputs = language_model(**inputs)
    logits = outputs.logits.cpu()

    idx_to_vocab = [a for a in tokenizer.vocab.keys()]
    indices = np.argpartition(logits.data.numpy()[0][mask_pos], range(tokenizer.vocab_size))
    pre_substitutes = []
    for i in range(80):
        index = indices[len(indices) - i - 1]
        token = idx_to_vocab[index]
        if token != word_to_replace:
            pre_substitutes.append(token)

    candidates = BERT_candidate_generation(word_to_replace, pre_substitutes, max_candidates, ps)
    return candidates


def BERT_candidate_generation(source_word, pre_tokens, num_selection, ps: PorterStemmer):
    cur_tokens = []

    source_stem = ps.stem(source_word)

    assert num_selection <= len(pre_tokens)

    for i in range(len(pre_tokens)):
        token = pre_tokens[i]

        if token[0:2] == "##":
            continue

        if (token == source_word):
            continue

        token_stem = ps.stem(token)

        if (token_stem == source_stem):
            continue

        if (len(token_stem) >= 3) and (token_stem[:3] == source_stem[:3]):
            continue

        cur_tokens.append(token)

        if (len(cur_tokens) == num_selection):
            break

    if (len(cur_tokens) == 0):
        cur_tokens = pre_tokens[0:num_selection + 1]

    assert len(cur_tokens) > 0

    return cur_tokens


def preprocess_SR(source_word, substitution_selection, ranker: Ranker):
    ss = []
    ##ss_score=[]
    sis_scores=[]
    count_scores=[]

    isFast = True

    if(source_word not in ranker.fasttext_dico):
        isFast = False
    else:
        source_emb = ranker.fasttext_emb[ranker.fasttext_dico.index(source_word)].reshape(1,-1)

    #ss.append(source_word)

    for sub in substitution_selection:

        if sub not in ranker.word_count:
            continue
        else:
            sub_count = ranker.word_count[sub]

        if(sub_count<=3):
            continue

        #if sub_count<source_count:
         #   continue
        if isFast:
            if sub not in ranker.fasttext_dico:
                continue

            token_index_fast = ranker.fasttext_dico.index(sub)
            sis = cosine(source_emb, ranker.fasttext_emb[token_index_fast].reshape(1,-1))

            #if sis<0.35:
            #    continue
            sis_scores.append(sis)

        ss.append(sub)
        count_scores.append(sub_count)

    return ss,sis_scores,count_scores


def cross_entropy_word(X, i, pos):
    # print(X)
    # print(X[0,2,3])
    X = softmax(X, axis=1)
    loss = 0
    loss -= np.log10(X[i, pos])
    return loss


def get_score(sentence, tokenizer, maskedLM):
    tokenize_input = tokenizer.tokenize(sentence)

    len_sen = len(tokenize_input)

    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'

    tokenize_input.insert(0, START_TOKEN)
    tokenize_input.append(SEPARATOR_TOKEN)

    input_ids = tokenizer.convert_tokens_to_ids(tokenize_input)

    # tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    # print("tensor_input")
    # print(tensor_input)
    # tensor_input = tensor_input.to('cuda')
    sentence_loss = 0

    for i, word in enumerate(tokenize_input):

        if (word == START_TOKEN or word == SEPARATOR_TOKEN):
            continue

        orignial_word = tokenize_input[i]
        tokenize_input[i] = '[MASK]'
        # print(tokenize_input)
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        # print(mask_input)
        mask_input = mask_input.to('cuda')
        with torch.no_grad():
            result = maskedLM(mask_input)
            att = result.attentions
            pre_word = result.logits
        word_loss = cross_entropy_word(pre_word[0].cpu().numpy(), i, input_ids[i])
        sentence_loss += word_loss
        # print(word_loss)
        tokenize_input[i] = orignial_word

    return np.exp(sentence_loss / len_sen)


def LM_score(source_word, source_sentence, substitution_selection, tokenizer, maskedLM):
    # source_index = source_context.index(source_word)

    source_sentence = source_sentence.strip()
    # print(source_sentence)
    LM = []

    source_loss = get_score(source_sentence, tokenizer, maskedLM)

    for substibution in substitution_selection:
        sub_sentence = source_sentence.replace(source_word, substibution)

        # print(sub_sentence)
        score = get_score(sub_sentence, tokenizer, maskedLM)

        # print(score)
        LM.append(score)

    return LM, source_loss


def substitution_ranking(
    source_word, source_sentence, substitution_selection, ranker: Ranker, tokenizer, maskedLM
) -> List[str]:

    ss, sis_scores, count_scores = preprocess_SR(source_word, substitution_selection, ranker)

    # print(ss)
    if len(ss) == 0:
        return [source_word]

    if len(sis_scores) > 0:
        seq = sorted(sis_scores, reverse=True)
        sis_rank = [seq.index(v) + 1 for v in sis_scores]

    rank_count = sorted(count_scores, reverse=True)

    count_rank = [rank_count.index(v) + 1 for v in count_scores]

    lm_score, source_lm = LM_score(source_word, source_sentence, ss, tokenizer, maskedLM)

    rank_lm = sorted(lm_score)
    lm_rank = [rank_lm.index(v) + 1 for v in lm_score]

    bert_rank = []
    ppdb_rank = []
    cgPPDB = ranker.ppdb.predict(source_word)
    for i in range(len(ss)):
        bert_rank.append(i + 1)

        if ss[i] in cgPPDB:
            ppdb_rank.append(1)
        else:
            ppdb_rank.append(len(ss) / 3)

    if len(sis_scores) > 0:
        all_ranks = [bert + sis + count + LM + ppdb for bert, sis, count, LM, ppdb in zip(bert_rank, sis_rank, count_rank, lm_rank, ppdb_rank)]
    else:
        all_ranks = [bert + count + LM + ppdb for bert, count, LM, ppdb in zip(bert_rank, count_rank, lm_rank, ppdb_rank)]
    # all_ranks = [con for con in zip(context_rank)]

    candidates_with_rank = zip(substitution_selection, all_ranks)
    ranked_candidates = [a[0] for a in sorted(candidates_with_rank, key=lambda x: x[1])]

    return ranked_candidates

    # pre_index = all_ranks.index(min(all_ranks))
    #
    # # return ss[pre_index]
    #
    # pre_count = count_scores[pre_index]
    #
    # if source_word in ranker.word_count:
    #     source_count = ranker.word_count[source_word]
    # else:
    #     source_count = 0
    #
    # pre_lm = lm_score[pre_index]
    #
    # # print(lm_score)
    # # print(source_lm)
    # # print(pre_lm)
    #
    # # pre_word = ss[pre_index]
    #
    # if source_lm > pre_lm or pre_count > source_count:
    #     pre_word = ss[pre_index]
    # else:
    #     pre_word = source_word
    #
    # return pre_word


def _load_models() -> tuple:

    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', do_lower_case=True)
    language_model = BertForMaskedLM.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', output_attentions=True)
    language_model.to(device)
    print(datetime.now() - start_time)

    ps = PorterStemmer()
    ranker = Ranker()
    ranker.read_features(
        EMBEDDINGS_FILE_PATH,
        PPDB_FILE_PATH,
    )
    return tokenizer, language_model, ps, ranker, device

def _arrange_for_output(candidates: List[RankedWordCandidate]) -> List[List[str]]:
    output = []
    last_rank = -1
    rank_list = []
    for candidate in candidates:
        if last_rank != candidate.merged_rank:
            rank_list = []
            output.append(rank_list)
        rank_list.append(candidate.word)
    return output


def _format_line(i: int, result_list: List[List[str]]) -> str:
    formatted_results = ' '.join(['{' + ', '.join(result) + '}' for result in result_list])
    return f"Sentence {i} rankings: {formatted_results}"


def measure_efficiency():

    file_size = path.getsize(EMBEDDINGS_FILE_PATH)
    file_size += path.getsize(PPDB_FILE_PATH)
    file_size += path.getsize(BERT_FILE_PATH)

    disk_usage = file_size / 1024 / 1024
    start_time = datetime.now()
    start_gpu_memory = get_gpu_memory()
    start_cpu_memory = get_cpu_memory()
    tokenizer, language_model, ps, ranker, device = _load_models()
    end_cpu_memory = get_cpu_memory()
    end_gpu_memory = get_gpu_memory()
    end_time = datetime.now()
    cpu_usage = end_cpu_memory - start_cpu_memory
    gpu_usage = start_gpu_memory - end_gpu_memory
    load_processing_time = end_time - start_time
    start_time = datetime.now()
    run_horn_for_efficiency(tokenizer, language_model, ps, ranker, device)
    end_time = datetime.now()
    elapsed_processing_time = end_time - start_time
    stats = EfficiencyStats(disk_usage, cpu_usage, gpu_usage, load_processing_time.seconds, elapsed_processing_time.seconds)
    print_stats("bertls_wwm", stats, stats)


def run_horn_for_efficiency(tokenizer, language_model, ps, ranker, device):
    lexmturk_file = LexMturkFile.load("data/horn/lex.mturk.es_10.txt")
    max_candidates = 30

    horn_candidates_file_path = "data/horn/results/temp_10_candidates.txt"

    with open(horn_candidates_file_path, "w") as horn_candidates_file:
        for line in lexmturk_file:
            candidates = candidates_generation(line.sentence, line.target_word, max_candidates, tokenizer, language_model, ps, device)
            ranked = substitution_ranking(line.target_word, line.sentence, candidates, ranker, tokenizer, language_model)
            line = "\t".join(ranked)
            horn_candidates_file.write(f"{line}\n")


def run_horn():
    tokenizer, language_model, ps, ranker, device = _load_models()
    horn_lex_mturk_file_path = "data/horn/lex.mturk.es.txt"
    horn_candidates_file_path = "data/horn/results/bertls_candidates.txt"
    horn_scores_file_path = "data/horn/results/bertls_scores.txt"

    lexmturk_file = LexMturkFile.load(horn_lex_mturk_file_path)
    max_candidates = 30
    with open(horn_candidates_file_path, "w") as horn_candidates_file:
        for line in lexmturk_file:
            candidates = candidates_generation(line.sentence, line.target_word, max_candidates, tokenizer, language_model, ps, device)
            ranked = substitution_ranking(line.target_word, line.sentence, candidates, ranker, tokenizer, language_model)
            line = "\t".join(ranked)
            horn_candidates_file.write(f"{line}\n")
    with open(horn_scores_file_path, 'w') as results_file:
        for n_candidates in (1, 2, 3, 4, 5, 10):
            precision, accuracy, changed = horn_scorer.score(horn_lex_mturk_file_path, horn_candidates_file_path, n_candidates)
            results_file.write("\n")
            results_file.write(f"n-best:    {n_candidates}\n")
            results_file.write(f"Precision: {precision:.4f}\n")
            results_file.write(f"Accuracy:  {accuracy:.4f}\n")
            results_file.write(f"Changed:   {changed:.4f}\n")


if __name__ == "__main__":

    measure_efficiency()
    # run_horn()
