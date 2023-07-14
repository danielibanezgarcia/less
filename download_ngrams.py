import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import path, popen, system
from pathlib import Path
from typing import Optional
from tqdm import tqdm

MAX_PROCESSES = 5


def process_ngram_files(ngram_size: int, n_files: int) -> None:
    tgt_file_path = f"en-{ngram_size}-grams.csv"
    Path(tgt_file_path).touch()
    process_ngram_file(ngram_size, 1, n_files, tgt_file_path)
    # with tqdm(total=n_files) as pbar:
    #     with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
    #         futures = [executor.submit(process_ngram_file, ngram_size, i, n_files, tgt_file_path)
    #                    for i in range(n_files)]
    #         for _ in tqdm(as_completed(futures)):
    #             pbar.update(1)
    print()


def process_ngram_file(ngram_size: int, i: int, n_files: int, tgt_file_path: str) -> None:
    file_name = f"{ngram_size}-{i:0>5}-of-{n_files:0>5}"
    print(f"Processing file {file_name}")
    remote_src_file_path = f"http://storage.googleapis.com/books/ngrams/books/20200217/eng-gb/{file_name}.gz"
    zipped_src_file_path = f"/home/daniel/clients/readableai/{file_name}.gz"
    src_file_path = f"/home/daniel/clients/readableai/{file_name}"
    print("Downloading...")
    if not path.isfile(zipped_src_file_path):
        system(f"curl {remote_src_file_path} --output {zipped_src_file_path}")
    print("Downloaded")
    print("Unzipping...")
    if not path.isfile(src_file_path):
        system(f"gzip -d {zipped_src_file_path}")
    print("Unzipped")
    print("Extracting...")
    extract_file(src_file_path, tgt_file_path)
    print("Extracted")
    print("Removing...")
    system(f"rm {src_file_path}")
    print("Removed")


def extract_file(src_file_path: str, tgt_file_path: str):
    lines = count_lines_in_file(src_file_path)
    with open(src_file_path, 'r', encoding="utf-8") as src_file, open(tgt_file_path, 'a', encoding="utf-8") as target_file:
        csv_reader = csv.reader(src_file, delimiter="\t", quoting=csv.QUOTE_NONE)
        for src_row in tqdm(csv_reader, total=lines):
            ngram = process_ngram(src_row[0])
            if ngram is None:
                continue
            src_years_data = {int(year_data[:4]): int(year_data.split(",")[1]) for year_data in src_row[1:]}
            target_years = sum([src_years_data.get(year, 0) for year in range(1990, 2020)])
            if target_years < 30:
                continue
            target_row = f"{ngram}\t{target_years}\n"
            target_file.write(target_row)


def count_lines_in_file(file_path: str) -> int:
    return int(popen(f"wc -l {file_path}").read().split()[0])


CONTAINS_DIGITS = re.compile("[0-9]")
CONTAINS_UNDERSCORE = re.compile("_")
CONTAINS_LETTERS = re.compile("[a-zA-Z]")


def process_ngram(ngram: str) -> Optional[str]:
    if re.search(CONTAINS_UNDERSCORE, ngram):
        return None
    if re.search(CONTAINS_DIGITS, ngram):
        return None
    if not re.search(CONTAINS_LETTERS, ngram):
        return None
    return ngram


if __name__ == "__main__":

    process_ngram_files(2, 118)
    # process_ngram_files(3, 688)


