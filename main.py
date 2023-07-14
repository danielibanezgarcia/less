import csv
import pickle

from wordfreq import word_frequency, get_frequency_dict

from less.common.util import get_window_context

if __name__ == "__main__":

    context_before_word = ['a', 'b', 'c']
    context_after_word = ['e', 'f', 'g']
    context_before_word = context_before_word[-min(len(context_before_word), 5):]
    context_after_word = context_after_word[:min(len(context_after_word), 5)]
    context = []
    for word in context_before_word + context_after_word:
        context.append(word)

    print()
    with open(r"2-grams.pkl", "rb") as input_file:
        bigrams = pickle.load(input_file)

    print(bigrams.get("compositor alemán", 0))
    print(bigrams.get("compositor vienés", 0))
    print(bigrams.get("compositor folclórico", 0))
    print(bigrams.get("compositor folklórico", 0))

    # with open('/home/daniel/clients/readableai/3-00089-of-00688') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter='\t')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count == 0:
    #             print(f'Column names are {", ".join(row)}')
    #             line_count += 1
    #         else:
    #             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
    #             line_count += 1
    #     print(f'Processed {line_count} lines.')