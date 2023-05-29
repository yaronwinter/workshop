from lang_model import LanguageModel
from vocabulary import Vocabulary

class Ngram:

    __PRIME_FACTOR = 6091

    def __init__(self, word_ids: list):
        self._ids = word_ids

        val = 1
        for word_id in self._ids:
            val = Ngram.__PRIME_FACTOR * val + word_id
        self._hash_val = val

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, k: int) -> int:
        return self._ids[k]

    def __iter__(self):
        return self._ids

    def __eq__(self, other) -> bool:
        n = len(self._ids)
        if len(other._ids) != n:
            return False

        if self._hash_val != other._hash_val:
            return False

        for i in range(n):
            if self._ids[i] != other._ids[i]:
                return False

        return True

    def __lt__(self, other):
        n = len(self._ids)
        if n < len(other._ids):
            return True
        elif len(other._ids) < n:
            return False

        for i in range(n):
            diff = self._ids[i] - other._ids[i]
            if diff < 0:
                return True
            elif diff > 0:
                return False

        return False

    def __hash__(self) -> int:
        return self._hash_val

    def prefix(self):
        return Ngram(self._ids[0: -1])

    def suffix(self):
        return Ngram(self._ids[1: ])


class NgramLanguageModel(LanguageModel):

    ZERO_LOG_PROB = -99.0
    OOV_PENALTY = -0.5
    MIN_BACKOFF_WEIGHT = 0.001

    def __init__(self):
        super().__init__()
        self._ngram_weights = [None]
        self._ngram_backoffs = [None]
        self._order = 0
        self._oov_weight = 0.0

    def order(self) -> int:
        return self._order

    def count(self, n: int) -> int:
        if (n <= 0) or (n > self._order):
            return 0

        return len(self._ngram_weights[n])

    def ngrams(self, n: int):
        if (n <= 0) or (n > self._order):
            return []

        return sorted(self._ngram_weights[n].keys())

    def __contains__(self, ngram: Ngram) -> bool:
        n = len(ngram)
        if (n <= 0) or (n > self._order):
            return False

        return ngram in self._ngram_weights[n]

    def __getitem__(self, ngram: Ngram) -> tuple:
        n = len(ngram)
        weight = self._ngram_weights[n][ngram]

        backoff = 0
        if n < self._order and ngram in self._ngram_backoffs[n]:
            backoff = self._ngram_backoffs[n][ngram]

        return weight, backoff

    def score(self, words: list) -> float:
        # Nothing to do if the model is empty.
        if self._order == 0:
            return 0

        # Convert the input words sequence into a list of word IDs, padding it with a start ID and an end ID.
        word_ids = [Vocabulary.START_ID] + [self._vocabulary.id(word) for word in words] + [Vocabulary.END_ID]

        accm = 0
        for to_ind in range(2, len(word_ids)):
            from_ind = max(0, to_ind - self._order)
            accm += self.__ngram_weight(Ngram(word_ids[from_ind: to_ind]))

        return accm

    def score_suffix(self, word_ids: list) -> float:
        # Nothing to do if the model is empty.
        if self._order == 0:
            return 0

        to_ind = len(word_ids)
        from_ind = max(0, to_ind - self._order)
        return self.__ngram_weight(Ngram(word_ids[from_ind: to_ind]))

    def __ngram_weight(self, ngram: Ngram) -> float:
        # Look for the given n-gram.
        n = len(ngram)
        if ngram in self._ngram_weights[n]:
            # If the model contains the n-gram, just return its weight.
            return self._ngram_weights[n][ngram]
        elif n == 1:
            # In case of a non-existing unigram, return the OOV weight.
            return self._oov_weight

        # Compute the conditional weight of the suffix of order (n - 1), an add the backoff weight of the prefix.
        suffix = ngram.suffix()
        weight = self.__ngram_weight(suffix)

        prefix = ngram.prefix()
        if prefix in self._ngram_backoffs[n - 1]:
            weight += self._ngram_backoffs[n - 1][prefix]

        return weight

_ARPABO_HEADER_LINE = '\\data\\'
_ARPABO_NGRAM_PREFIX = 'ngram'
_ARPABO_EQUAL_SIGN = '='
_ARPABO_NGRAMS_SUFFIX = '-grams:'
_ARPABO_FOOTER_LINE = '\\end\\'

def read_arpabo(filename: str) -> NgramLanguageModel:
    lang_model = NgramLanguageModel()
    vocabulary = lang_model.vocabulary()

    read_header = False
    found_data = False
    ngram_counts = [0]
    with open(filename, 'r', encoding='utf-8-sig') as in_file:
        # Read the ARPABO header.
        while not read_header:
            line = in_file.readline()
            if line is None:
                raise(Exception('Failed to read the ARPABO header.'))

            line = line.strip()

            if not found_data:
                if line == _ARPABO_HEADER_LINE:
                    found_data = True

                if len(line) == 0:
                    continue
            else:
                if line.startswith(_ARPABO_NGRAM_PREFIX):
                    try:
                        fields = line[len(_ARPABO_NGRAM_PREFIX):].split(_ARPABO_EQUAL_SIGN)
                        n = int(fields[0].strip())
                        count = int(fields[1].strip())
                    except:
                        raise (Exception('Failed to read the ARPABO header.'))

                    if n != len(ngram_counts):
                        raise (Exception('Failed to read the ARPABO header.'))
                    ngram_counts.append(count)

                elif len(line) == 0:
                    read_header = True

        lang_model._order = len(ngram_counts) - 1

        # Read the n-grams.
        min_unigram_weight = 0
        for n in range(1, len(ngram_counts)):
            line = ''
            while len(line) == 0:
                line = in_file.readline()
                if line is None:
                    raise (Exception('Failed to read the ARPABO n-grams header of order %d.' % n))

                line = line.strip()

            expected_line = '\\%d%s' % (n, _ARPABO_NGRAMS_SUFFIX)
            if line != expected_line:
                raise (Exception('Failed to read the ARPABO n-grams header of order %d.' % n))

            curr_weights = {}
            curr_backoffs = {}
            count = 0
            while count < ngram_counts[n]:
                line = in_file.readline()
                if line is None:
                    raise (Exception('Failed to read the complete ARPABO n-grams of order %d (expected %d n-grams, EOL after %d.' %
                                     (n, ngram_counts[n], count)))

                line = line.strip()
                if len(line) == 0:
                    continue

                if line.find('\t') > 0:
                    line = line.replace('\t', ' ')

                fields = line.split(' ')
                if len(fields) < n + 1:
                    raise (Exception('Failed to read the ARPABO n-gram line of order %d: "%s".' % (n, line)))

                try:
                    weight = float(fields[0])
                except:
                    raise (Exception('Failed to read the ARPABO n-gram line of order %d: "%s".' % (n, line)))

                word_ids = [0] * n

                if n == 1:
                    word = fields[1]
                    if word in vocabulary:
                        word_ids[0] = vocabulary.id(word)
                        if word_ids[0] > Vocabulary.OOV_ID:
                            raise (Exception('Duplicate word %s found among the unigrams.' % word))
                    else:
                        word_ids[0] = vocabulary.add(word)[0]

                    if weight < min_unigram_weight:
                        min_unigram_weight = weight
                else:
                    for k in range(n):
                        word = fields[k + 1]
                        if word in vocabulary:
                            word_ids[k] = vocabulary.id(word)
                        else:
                            raise (Exception('The word %s is out of the unigram vocabulary.' % word))

                ngram = Ngram(word_ids)
                curr_weights[ngram] = weight

                if len(fields) == n + 2:
                    try:
                        backoff = float(fields[n + 1])
                    except:
                        raise (Exception('Failed to read the ARPABO n-gram line of order %d: "%s".' % (n, line)))
                    curr_backoffs[ngram] = backoff

                count += 1

            lang_model._ngram_weights.append(curr_weights)
            lang_model._ngram_backoffs.append(curr_backoffs)

        # Read the ARPABO footer line.
        line = ''
        while len(line) == 0:
            line = in_file.readline()
            if line is None:
                raise (Exception('Failed to read the ARPABO n-grams header of order %d.' % n))

            line = line.strip()

        if line != _ARPABO_FOOTER_LINE:
            raise (Exception('Failed to read the ARPABO footer line.'))

    lang_model._oov_weight = min_unigram_weight + NgramLanguageModel.OOV_PENALTY

    return lang_model


def write_arpabo(lang_model: NgramLanguageModel, filename: str):

    vocabulary = lang_model.vocabulary()
    model_order = lang_model.order()

    with open(filename, 'w', encoding='utf-8-sig') as out_file:
        # Write the ARPABO header.
        out_file.write('%s\n' % _ARPABO_HEADER_LINE)

        for n in range(1, model_order + 1):
            out_file.write('%s %d%s%d\n' % (_ARPABO_NGRAM_PREFIX, n, _ARPABO_EQUAL_SIGN, lang_model.count(n)))
        out_file.write('\n')

        # Go over all n-gram orders.
        for n in range(1, model_order + 1):
            # Write the n-grams of the current order n.
            out_file.write('\\%d%s\n' % (n, _ARPABO_NGRAMS_SUFFIX))

            for ngram in lang_model.ngrams(n):
                weight, backoff = lang_model[ngram]

                out_file.write('%f' % weight)
                for k in range(n):
                    out_file.write(' %s' % vocabulary[ngram[k]])

                if n < model_order and backoff != 0:
                    out_file.write(' %f' % backoff)
                out_file.write('\n')

            out_file.write('\n')

        # Write the ARPABO footer.
        out_file.write('%s\n' % _ARPABO_FOOTER_LINE)
