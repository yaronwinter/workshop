import math

from vocabulary import Vocabulary
from ngram_lang_model import Ngram, NgramLanguageModel
from math import log10


class NgramModelAccumulator:

    def __init__(self, model_order: int):
        self._order = model_order
        self._words = [None, Vocabulary.START_SYMBOL, Vocabulary.END_SYMBOL]
        self._word_map = {Vocabulary.END_SYMBOL: Vocabulary.END_ID}
        self._ngram_counts = [None]

        for n in range(model_order):
            empty_counter = {}
            self._ngram_counts.append(empty_counter)

    def order(self):
        return self._order

    def accumulate(self, sentence: list):
        # Convert the sentence to a list of word IDs.
        word_ids = [Vocabulary.START_ID]
        for word in sentence:
            if word in self._word_map:
                word_id = self._word_map[word]
            else:
                word_id = len(self._words)
                self._words.append(word)
                self._word_map[word] = word_id

            word_ids.append(word_id)

        word_ids.append(Vocabulary.END_ID)

        # Go over all n-gram orders (unigrams first, then bigrams, etc.).
        for n, counter in enumerate(self._ngram_counts):
            if n == 0:
                continue

            # Go over all n-grams of order n and count them.
            for end_ind in range(n, len(word_ids) + 1):
                start_ind = end_ind - n
                ngram = Ngram(word_ids[start_ind: end_ind])

                if ngram in counter:
                    counter[ngram] += 1
                else:
                    counter[ngram] = 1


class NgramModelBuilder:

    def __init__(self, accum: NgramModelAccumulator):
        self._accum = accum
        self._min_counts = []
        self._ngram_counts = [None]
        self._total_unigram_count = 0
        self._ngram_prefix_counts = [None]
        self._ngram_backoff_probs = [None]
        self._ngram_discounted_probs = {}

    def build(self, min_count) -> NgramLanguageModel:
        # Determine the minimal number of occurrences for each n-gram order.
        model_order = self._accum.order()

        self._min_counts = [0] * (model_order + 1)
        if type(min_count) is int and min_count > 0:
            for n in range(1, model_order + 1):
                self._min_counts[n] = min_count

        elif type(min_count) is list:
            n = 1
            for count in min_count:
                self._min_counts[n] = count
                n += 1

            for n in range(1, model_order + 1):
                if self._min_counts[n] < self._min_counts[n - 1]:
                    self._min_counts[n] = self._min_counts[n - 1]

        # Initialize the output language model and set its vocabulary.
        # This function also sets the internal _ngram_counts dictionaries.
        lang_model = self.__set_vocabulary()

        # Compute the total number of unigram occurrences.
        total_count = 0

        for unigram, count in self._ngram_counts[1].items():
            # Do not count the <s> symbol.
            if unigram[0] != Vocabulary.START_ID:
                total_count += count

        self._total_unigram_count = total_count

        # Go over higher order n-grams, and compute the number of seen n-gram that each prefix has.
        for n in range(2, model_order + 1):
            prefix_counter = {}
            for ngram, count in self._ngram_counts[n].items():
                prefix = ngram.prefix()
                if prefix in prefix_counter:
                    prefix_counter[prefix] += 1
                else:
                    prefix_counter[prefix] = 1

            self._ngram_prefix_counts.append(prefix_counter)

        # Now process all n-grams, starting from the unigrams, then bigrams, and so on ...
        min_unigram_weight = 0
        for n in range(1, model_order + 1):
            # Compute the statistics needed for estimating the backoff weights.
            if n < model_order:
                self.__compute_backoff_probabilities(n)

            # Compute the weight and backoff (if relevant) for each n-gram of the current order.
            curr_min_count = self._min_counts[n]
            curr_weights = {}
            curr_backoffs = {}

            for ngram, count in self._ngram_counts[n].items():
                if count < curr_min_count:
                    continue
                    
                weight, backoff = self.__process_ngram(ngram, count)
                curr_weights[ngram] = weight
                if n < model_order and abs(backoff) > NgramLanguageModel.MIN_BACKOFF_WEIGHT:
                    curr_backoffs[ngram] = backoff

                if n == 1 and weight < min_unigram_weight:
                    min_unigram_weight = weight

            lang_model._ngram_weights.append(curr_weights)
            lang_model._ngram_backoffs.append(curr_backoffs)

        # Set the rest of the language-model properties.
        lang_model._oov_weight = weight + NgramLanguageModel.OOV_PENALTY

        # Reset all internal work areas.
        self._min_counts = []
        self._ngram_counts = [None]
        self._total_unigram_count = 0
        self._ngram_prefix_counts = [None]
        self._ngram_backoff_probs = [None]
        self._ngram_discounted_probs = {}

        return lang_model

    def __set_vocabulary(self) -> NgramLanguageModel:
        # Collect all unigrams whose number of occurrences is above the minimum.
        min_unigram_count = self._min_counts[1]
        source_unigram_counter = self._accum._ngram_counts[1]
        vocab_words = []
        max_id = 0

        for word, word_id in self._accum._word_map.items():
            count = source_unigram_counter[Ngram([word_id])]
            if word_id > Vocabulary.OOV_ID and count >= min_unigram_count:
                vocab_words.append(word)

            if word_id > max_id:
                max_id = word_id

        # Create the language-model vocabulary.
        lang_model = NgramLanguageModel()
        lang_model._order = self._accum.order()

        model_vocab = lang_model.vocabulary()

        # Prepare a mapping of the word IDs in the accumulator onto vocabulary word IDs.
        # Note that words with less than min_unigram occurrences are mapped onto an OOV word.
        vocab_ids_map = [Vocabulary.OOV_ID] * (max_id + 1)

        vocab_ids_map[Vocabulary.INVALID_ID] = Vocabulary.INVALID_ID
        vocab_ids_map[Vocabulary.START_ID] = Vocabulary.START_ID
        vocab_ids_map[Vocabulary.END_ID] = Vocabulary.END_ID

        for word in sorted(vocab_words):
            source_id = self._accum._word_map[word]
            target_id, _ = model_vocab.add(word)
            vocab_ids_map[source_id] = target_id

        # Convert the n-gram counters from the accumulator.
        model_order = self._accum.order()

        for n in range(1, model_order + 1):
            target_ngram_counter = {}

            for source_ngram, count in self._accum._ngram_counts[n].items():
                target_ids = [0] * n
                for k in range(n):
                    target_ids[k] = vocab_ids_map[source_ngram[k]]

                ngram = Ngram(target_ids)
                if ngram in target_ngram_counter:
                    target_ngram_counter[ngram] += count
                else:
                    target_ngram_counter[ngram] = count

            self._ngram_counts.append(target_ngram_counter)

        return lang_model

    def __compute_backoff_probabilities(self, n: int):
        # Go over all n-grams of order (n + 1) and accumulate the statistics for their prefix n-grams.
        prefix_stats = {}
        for ngram, count in self._ngram_counts[n + 1].items():
            # Get the discounted probabilities for the current (n + 1)-gram and for its suffix (of order n).
            ngram_prob = self.__discounted_probability(ngram)
            suffix_prob = self.__discounted_probability(ngram.suffix())

            # Accumulate the numerator and denominator for the prefix n-gram.
            prefix = ngram.prefix()
            if prefix not in prefix_stats:
                prefix_stats[prefix] = [ngram_prob, suffix_prob]
            else:
                stats = prefix_stats[prefix]
                stats[0] += ngram_prob
                stats[1] += suffix_prob

        # Calculate the backoff probabilities from the statistics we have just computed.
        curr_backoff_probs = {}
        for prefix, stats in prefix_stats.items():
            numer = 1.0 - stats[0]
            denom = 1.0 - stats[1]

            if denom < 1e-10:
                continue

            curr_backoff_probs[prefix] = numer / denom

        self._ngram_backoff_probs.append(curr_backoff_probs)

    def __discounted_probability(self, ngram: Ngram, do_cache: bool = True) -> float:
        n = len(ngram)
        count = self._ngram_counts[n][ngram]

        if n == 1:
            # No discounting performed on unigrams.
            return count / self._total_unigram_count

        # Check if the discounted probability is already computed and cached.
        if ngram in self._ngram_discounted_probs:
            return self._ngram_discounted_probs[ngram]

        # Perform the Witten-Bell discounting:
        prefix = ngram.prefix()
        suffix = ngram.suffix()
        num_prefix_ngrams = self._ngram_prefix_counts[n - 1][prefix]
        numer = count + num_prefix_ngrams * self.__discounted_probability(suffix)
        denom = self._ngram_counts[n - 1][prefix] + num_prefix_ngrams
        prob = numer / denom

        # Cache the result before returning it.
        if do_cache:
            self._ngram_discounted_probs[ngram] = prob

        return prob

    def __process_ngram(self, ngram: Ngram, count: int) -> tuple:
        n = len(ngram)
        model_order = self._accum.order()

        # Compute the discounted probability for the n-gram (without caching it).
        prob = self.__discounted_probability(ngram, False)

        if n > 1:
            # Interpolate with the back-off probability.
            prefix = ngram.prefix()
            suffix = ngram.suffix()

            suffix_prob = self.__discounted_probability(suffix)

            if prefix in self._ngram_backoff_probs[n - 1]:
                backoff_prob = self._ngram_backoff_probs[n - 1][prefix] * suffix_prob

                if (prob + backoff_prob) <= 1.0:
                    prob += backoff_prob

        # Compute the log probability of the n-gram.
        if n == 1 and ngram[0] == Vocabulary.START_ID:
            # Special treatment for the <s> unigram.
            log_prob = NgramLanguageModel.ZERO_LOG_PROB
        else:
            log_prob = math.log10(prob)

        # In case n is less than the model order, compute the back-off weight as well.
        backoff_weight = 0;

        if n < model_order:
            if ngram in self._ngram_backoff_probs[n]:
                backoff_prob = self._ngram_backoff_probs[n][ngram]
                backoff_weight = math.log10(backoff_prob)

                # Avoid positive or very small back-off weights.
                if backoff_weight > - NgramLanguageModel.MIN_BACKOFF_WEIGHT:
                    backoff_weight = 0

        return log_prob, backoff_weight
