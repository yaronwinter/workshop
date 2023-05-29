import re
from tqdm import tqdm
from emagine.viterbi import ngram_lang_model
from emagine.viterbi.ngram_model_builder import NgramModelAccumulator, NgramModelBuilder

MODEL_MIN_COUNTS = [1, 2, 4, 8, 16]

def prepare_train_set(raw_train_set: str, formatted_train_set: str):
    with open(raw_train_set, "r", encoding='utf-8') as f:
        raw_lines = [x.strip() for x in f.readlines()]

    with open(formatted_train_set, "w", encoding='utf-8') as f:
        for line in tqdm(raw_lines):
            f.write(re.sub(r"[^a-z]", "", line) + "\n")
            f.flush()

def train_lm(train_set: str, lm_file: str, ngram_order: int):
    with open(train_set, "r", encoding='utf-8') as f:
        train_lines = [x.strip() for x in f.readlines()]

    lm_accum = NgramModelAccumulator(ngram_order)
    for text in tqdm(train_lines):
        if len(text) == 0:
            continue
        lm_accum.accumulate(text)

    lm_builder = NgramModelBuilder(lm_accum)
    lang_model = lm_builder.build(MODEL_MIN_COUNTS)
    ngram_lang_model.write_arpabo(lang_model, lm_file)
