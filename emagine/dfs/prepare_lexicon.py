import pandas as pd
import re

# Number of top frequency, per length, words to leave
# in the lexicon.
VALID_LENGTH_AMOUNTS = {
    2: 50,
    3: 400,
    4: 1400,
    5: 4000,
    6: 5000,
    7: 5500
}
MAX_TESTED_LENGTH = 7
MAX_LEX_WORDS = 35000

def prepare_lexicon(raw_lexicon_file, clean_lexicon_file):
    with open(raw_lexicon_file, "r", encoding="utf-8") as f:
        raw_words = [x.strip() for x in f.readlines()]
    print("#raw words = " + str(len(raw_words)))

    formatted_words = [re.sub(r"[^a-z]", "", word) for word in raw_words]
    formatted_words = [word for word in formatted_words if len(word) > 0]
    print("#formatted words = " + str(len(formatted_words)))

    clean_words = [word for word in formatted_words if len(word) > 1 or word == "a" or word == "i"]
    print("#clean unigrams words = " + str(len(clean_words)))

    max_length = max([len(word) for word in clean_words])
    print("max length = " + str(max_length))
    for i in range(2, max_length + 1):
        current_amount = (VALID_LENGTH_AMOUNTS[MAX_TESTED_LENGTH] if i >= MAX_TESTED_LENGTH else VALID_LENGTH_AMOUNTS[i])
        clean_words = filter_words(clean_words, i, current_amount)
        print("#Words after filtering " + str(i) + " length: " + str(len(clean_words)))

    with open(clean_lexicon_file, "w", encoding="utf-8") as f:
        for word in clean_words[:MAX_LEX_WORDS]:
            f.write(word + "\n")
            f.flush()


def filter_words(words: list, current_length: int, current_amount: int) -> list:
    filtered_words = []
    num_words_in_length = 0
    for word in words:
        if len(word) != current_length:
            filtered_words.append(word)
        elif num_words_in_length < current_amount:
            filtered_words.append(word)
            num_words_in_length += 1
        else:
            continue
    return filtered_words
