from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors as EmbeddedModel
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
import utils.config as params

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print("Epoch #{} Start".format(self.epoch))
        
    def on_epoch_end(self, model):
        print("Epoch #{} End".format(self.epoch))
        self.epoch += 1

def train_word_embedding(train_corpus: str, config: dict) -> EmbeddedModel:
    sentences = LineSentence(datapath(train_corpus))
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


def load_embedding_model(model_path: str) -> EmbeddedModel:
    w2v_model = KeyedVectors.load_word2vec_format(datapath(model_path))
    w2v_model.init_sims()
    return w2v_model

def save_model(model: EmbeddedModel, file_path: str):
    model.wv.save_word2vec_format(file_path)
