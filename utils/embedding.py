from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
import utils.config as params
import numpy as np
from tqdm import tqdm

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print("Epoch #{} Start".format(self.epoch))
        
    def on_epoch_end(self, model):
        print("Epoch #{} End".format(self.epoch))
        self.epoch += 1

class Embedded_Words:
    def __init__(self, model_file: str, added_pads: list, norm: bool) -> None:
        self.vectors, self.w2i, self.i2w = self.read_model(model_file, added_pads, norm)

    def read_model(self, model_file: str, added_pads: list, norm: bool) -> tuple:
        with open(model_file, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines()]

        print(model_file)
        print(len(lines))
        print(lines[0])

        num_word, dim = [int(x) for x in lines[0].split()]
        vectors = np.zeros((num_word + len(added_pads), dim))
        w2i = {}
        i2w = {}
        for line in tqdm(lines[1:]):
            tokens = line.split()
            word = tokens[0]
            word_index = len(w2i)
            v = np.array([float(x) for x in tokens[1:]])
            if norm:
                v = v / np.linalg.norm(v)
            vectors[word_index] = v
            w2i[word] = word_index
            i2w[word_index] = word

        for word in added_pads:
            word_index = len(w2i)
            w2i[word] = word_index
            i2w[word_index] = word
        
        return vectors, w2i, i2w

def train_word_embedding(config: dict):
    sentences = LineSentence(datapath(config[params.EMBED_TRAIN_CORPUS]))
    logger = EpochLogger()
    w2v_model = Word2Vec(sentences,
                        vector_size=config[params.EMBED_VEC_DIM],
                        window=config[params.EMBED_WIN_WIDTH],
                        min_count=config[params.EMBED_MIN_COUNT],
                        workers=config[params.EMBED_NUM_WORKERS],
                        sg=config[params.EMBED_SKIP_GRAMS],
                        negative=config[params.EMBED_NUM_NEGS],
                        ns_exponent=config[params.EMBED_NS_EXP],
                        cbow_mean=config[params.EMBED_CBOW_MEAN],
                        epochs=config[params.EMBED_NUM_EPOCHS],
                        callbacks=[logger])
    
    return w2v_model


def load_embedding_model(model_path: str):
    return KeyedVectors.load_word2vec_format(datapath(model_path))

def load_and_add(config: dict, added_pads: list):
    model = load_embedding_model(config[params.EMBED_WORDS_FILE])
    if len(added_pads) == 0:
        model.fill_norms()
        return model

    added_vecs = [np.random.random(model.vector_size) for i in range(len(added_pads))]
    model.add_vectors(added_pads, added_vecs)
    model.fill_norms()
    return model

def save_model(model, file_path: str):
    model.wv.save_word2vec_format(file_path)
