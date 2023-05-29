from vocabulary import Vocabulary

class LanguageModel:

    def __init__(self):
        self._vocabulary = Vocabulary()

    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def score(self, words: list) -> float:
        pass
