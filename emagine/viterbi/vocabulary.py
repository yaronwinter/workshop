class Vocabulary:
    INVALID_ID = 0
    START_ID = 1
    END_ID = 2
    OOV_ID = 3

    START_SYMBOL = '<s>'
    END_SYMBOL = '</s>'
    OOV_SYMBOL = '<unk>'

    def __init__(self):
        self._words = [None, Vocabulary.START_SYMBOL, Vocabulary.END_SYMBOL, Vocabulary.OOV_SYMBOL]
        self._ids_map = {Vocabulary.START_SYMBOL : Vocabulary.START_ID,
                         Vocabulary.END_SYMBOL : Vocabulary.END_ID,
                         Vocabulary.OOV_SYMBOL : Vocabulary.OOV_ID}
        self._max_id = Vocabulary.OOV_ID

    def __contains__(self, word: str) -> bool:
        return word in self._ids_map

    def __getitem__(self, word_id: int) -> str:
        return self._words[word_id]

    def max_id(self):
        return self._max_id

    def id(self, word: str) -> int:
        if word in self._ids_map:
            return self._ids_map[word]
        else:
            return Vocabulary.OOV_ID

    def add(self, word: str) -> tuple:
        if word in self._ids_map:
            return self._ids_map[word], False

        self._words.append(word)
        self._max_id += 1
        self._ids_map[word] = self._max_id
        return self._max_id, True
