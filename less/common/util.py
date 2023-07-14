import csv
import json
import ntpath
import uuid
from glob import glob
from os import makedirs, path, popen
from string import punctuation
from typing import Dict, List, Tuple
from zipfile import ZipFile

import yaml

DEFAULT_ENCODING = "utf-8"


def read_dict_from_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding=DEFAULT_ENCODING) as file:
        return json.loads(file.read())


def write_dict_into_json_file(object: Dict, json_file_path: str, indent: int = 4):
    with open(json_file_path, "w", encoding=DEFAULT_ENCODING) as file:
        file.write(json.dumps(object, indent=indent))


def read_dict_from_yaml_file(yaml_file_path: str):
    with open(yaml_file_path, "r", encoding=DEFAULT_ENCODING) as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)


def write_dict_into_yaml_file(object: Dict, yaml_file_path: str):
    with open(yaml_file_path, "w", encoding=DEFAULT_ENCODING) as file:
        yaml.dump(object, file, default_flow_style=False)


def read_lines_from_txt_file(txt_file_path: str) -> List[str]:
    lines = []
    with open(txt_file_path, "r", encoding=DEFAULT_ENCODING) as file:
        for line in file:
            lines.append(line.rstrip())
    return lines


def read_txt_file(txt_file_path: str) -> str:
    with open(txt_file_path, "r", encoding=DEFAULT_ENCODING) as file:
        return file.read()


def save_txt_file(txt_file_path: str, content: str):
    with open(txt_file_path, "w", encoding=DEFAULT_ENCODING) as file:
        file.write(content)


def read_lines_from_csv_file(file_path: str, delimiter: str = ",") -> List[List[str]]:
    lines = []
    with open(file_path, 'r', encoding=DEFAULT_ENCODING) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        next(csv_reader, None)
        for row in csv_reader:
            lines.append(list(row))
    return lines


def read_files_content_in_folder(folder: str) -> Dict:
    request_folder_pattern = path.join(folder, "*")
    files = {}
    for file_path in glob(request_folder_pattern):
        files[file_path] = read_txt_file(file_path)
    return files


def write_lines_into_txt_file(lines: List[str], txt_file_path: str):
    with open(txt_file_path, "w", encoding=DEFAULT_ENCODING) as file:
        for line in lines:
            file.write("{}\n".format(line.rstrip()))


def extract_file_name(file_path: str) -> str:
    head, tail = ntpath.split(path.splitext(file_path)[0])
    return tail or ntpath.basename(head)


def ensure_directory_exists(dir_path: str):
    if path.isdir(dir_path):
        return
    makedirs(dir_path)


def unzip_files_to_folder(zip_file_path: str, target_folder: str):
    with ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(target_folder)


def get_extension(filename: str) -> str:
    last_dot_start = filename.rfind(".")
    if last_dot_start < 0:
        return ""
    return filename[last_dot_start + 1 :]


def create_uuid():
    return str(uuid.uuid4())


def get_ngrams(tokens: list, ngram_size: int) -> List[list]:
    ngrams = []
    for i in range(len(tokens) - ngram_size + 1):
        ngrams.append(tokens[i: i + ngram_size])
    return ngrams


def count_lines_in_file(file_path: str) -> int:
    return int(popen("wc -l {}".format(file_path)).read().split()[0])


def detokenize(tokenized_text: str, sep: str = ' ') -> str:
    words = tokenized_text.split(sep)
    text = words[0]
    for word in words[1:]:
        if "'" in word or word in punctuation:
            text += word
        else:
            text += ' ' + word
    return text


def normalize(value: float, minimum: float, maximum: float) -> float:
    return (value - minimum) / (maximum - minimum)


def get_window_context(context_before_word: List[str], context_after_word: List[str], ngram_size: int) -> Tuple[List[str], List[str]]:
    number_of_window_elements_at_each_side = ngram_size - 1
    start_element = -1 * min(number_of_window_elements_at_each_side, len(context_before_word))
    window_context_before_word = context_before_word[start_element:]
    end_element = min(number_of_window_elements_at_each_side, len(context_after_word))
    window_context_after_word = context_after_word[:end_element]
    return window_context_before_word, window_context_after_word
