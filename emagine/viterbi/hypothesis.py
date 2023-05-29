from ngram_lang_model import NgramLanguageModel, read_arpabo, Ngram
from vocabulary import Vocabulary

class Hypothesis:

	def __init__(self):
		self.len = 0
		self.permutation = []
		self.ordered_ids = []
		self.score = 0


	def extend(self, perm_index: int, word_ids: list, lang_model: NgramLanguageModel):
		ex_hypot = Hypothesis()
		ex_hypot.len = self.len + 1
		ex_hypot.permutation = self.permutation + [perm_index]

		delta_score = 0
		ex_hypot.ordered_ids = []
		for k, message in enumerate(self.ordered_ids):
			ex_message = message + [word_ids[k][perm_index]]
			delta_score += lang_model.score_suffix(ex_message)
			ex_hypot.ordered_ids.append(ex_message)

		ex_hypot.score = self.score + delta_score
		return ex_hypot


	def extend_final(self, lang_model):
		ex_hypot = Hypothesis()
		ex_hypot.len = self.len
		ex_hypot.permutation = self.permutation

		delta_score = 0
		ex_hypot.ordered_ids = []
		for k, message in enumerate(self.ordered_ids):
			ex_message = message + [Vocabulary.END_ID]
			delta_score += lang_model.score_suffix(ex_message)
			ex_hypot.ordered_ids.append(ex_message)

		ex_hypot.score = self.score + delta_score
		return ex_hypot
