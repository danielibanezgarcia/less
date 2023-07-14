import re
from os import path, system
import pickle


CONTAINS_LETTERS = re.compile("[a-zA-Z]")


def process_ngram_file(ngrams_file: str) -> None:
    print(f"Processing file {ngrams_file}")
    # max_ngrams = get_max_ngrams_since(1990, 2019)
    ngrams = build_dict(ngrams_file)
    persist_to_pickle(ngrams, ngrams_file)


def get_max_ngrams_since(initial_year: int, end_year: int) -> int:
    remote_totals_file_path = f"http://storage.googleapis.com/books/ngrams/books/20200217/spa/totalcounts-2"
    totals_file_path = path.basename(remote_totals_file_path)
    print("Downloading...")
    system(f"curl {remote_totals_file_path} --output {totals_file_path}")
    print("Downloaded")
    print("Extracting...")
    with open(totals_file_path, "r") as totals_file:
        totals = totals_file.read().split("\t")
    totals_in_range = [total for total in totals if len(total.strip()) > 0 and initial_year <= int(total[:4]) <= end_year]
    max_ngrams = sum([int(year_data.split(",")[1]) for year_data in totals_in_range])
    print("Extracted")
    print("Removing...")
    system(f"rm {totals_file_path}")
    print("Removed")
    return max_ngrams


def build_dict(ngrams_file_path: str) -> dict:
    bigrams = {}
    with open(ngrams_file_path, "r") as ngrams_file:
        i = 0
        for line in ngrams_file:
            i += 1
            if i % 1000000 == 0:
                print(i)
            if not re.search(CONTAINS_LETTERS, line):
                continue
            try:
                ngram, value = line.strip().split("\t")
                value = int(value)
            except Exception:
                continue
            bigrams[ngram] = value
    return bigrams


def persist_to_pickle(ngrams: dict, ngrams_file: str) -> None:
    pickle_file_path = ngrams_file[:ngrams_file.rindex(".")] + ".pkl"
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(ngrams, pickle_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    process_ngram_file("../less/data/ngrams/en-2-grams.csv")
    # process_ngram_file("2-grams.csv")


