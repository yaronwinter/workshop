from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print("Epoch #{} Start".format(self.epoch))
        
    def on_epoch_end(self, model):
        print("Epoch #{} End".format(self.epoch))
        self.epoch += 1

EMBED_PREC = "{:.7f}"
VEC_DIM = 200
WIN_WIDTH = 5
MIN_COUNT = 3
WORKERS = 3
SKIP_GRAMS = 1
NEGATIVES = 20
NS_EXP = 0.75
CBOW_MEAN = 1
EPOCHS = 5
NUM_STOP_WORDS = 25
MIN_FREQ = 1

def train_word_embedding(train_corpus):
    sentences = LineSentence(datapath(train_corpus))
    logger = EpochLogger()
    w2v_model = Word2Vec(sentences, vector_size=VEC_DIM, window=WIN_WIDTH,
                         min_count=MIN_COUNT, workers=WORKERS,
                         sg=SKIP_GRAMS, negative=NEGATIVES,
                         ns_exponent=NS_EXP, cbow_mean=CBOW_MEAN,
                         epochs=EPOCHS, callbacks=[logger])
    
    return w2v_model


def load_embedding_model(model_path):
    w2v_model = KeyedVectors.load_word2vec_format(datapath(model_path))
    w2v_model.init_sims()
    return w2v_model

def save_model(model, file_path):
    model.wv.save_word2vec_format(file_path)
