from hypothesis import Hypothesis
from vocabulary import Vocabulary
import ngram_lang_model
from tqdm import tqdm

PERMUTATIONS = [
    "IATSNEDBOONANYNOOOTSPROOLEDCCUODYOUFBOAHORRFWWKTHJEHDEENHUTISYNOCTEAFAOFLCSAONRLOIEUCHNATNULAAYDSUDMEIMATADUIMSYRHVOADETMMJTPIIN",
    "EOHYEYRHBOESOCTWUSSEVMHYLIDTHNOLEUSLEDAIRLOLEAEWAETEELKNBSEHTSNWNBNNETNMEELEMNHHRTTRIYIARMENCELBGSCIETEOEIEALESRAICETSLSHAELHIEL",
    "FSTEDTHGDESTPSSORRINFHNCCEAEEEEUEIARGREATEAHTTIEAEOSITNDHNCEINLUODTNHSXERASTIHITTSESRINTTAWEATAVDOAOOSILIEEHHCPYTNICMTTIITUEVTHO",
    "MCDAIERIBROOEDGEUDUIARRLEEENSPEEEIERENCFOETRDVRATSETRSCINLNMNEOENATOOLYBNURLAOOTSEAMEEAAEASECCEKRMCNNECRMSFPELSRRIEXTDHCSRNUDPRA",
    "AFTTEEGDOCISGEAHHRKNIEVTNNASEHOHBTTSNOEHAKTOADTHAETSHDIMEICMULOERHEEINUEOAINOTCSETNDSIOEBMVAKGWYYADADNOWTTUNTNMEEBNDAYADRSANSHRN",
    "EIERTDMHNMONAENIADLUDLGSALNISIHGLCTBARRNLEEHAREAPONEMTNWBSWEOLWYNHITOGOYEOEETLETBTDLEAHAOWOSTTEIOONRLOSETRINHAHHHWILSMPRYTIEWTSE",
    "HLABRSEOIAERRVDTNYAGAROWTILWNOGNAIINURGBROTHGETNDHIWODHINFIGONORRTFOOEHAAGWRESLRYSFENAREIEHLEOENUNRESONRTTWRAANISHTCANITPWMWWTVD",
    "STLIKNRTETDEEEIEHFTEPBCTHEAIALIWNTOROTRSIHHUIEAININNRRSEOTNNNETOODNHETSGPLTEXUIHILTIDAUSFHNTOTREITEGMHANSAHGGRRYUIEBMADFRCSEDDTB",
    "ILHHHRTDFPBEGOREEIXLWKTWCEDEIABTSGOOANIUFRDHAOFDSNCNHDTWVNTLERLNERASRAAUBVSSEEUPCSTENYDMELOWFTEEBAUEOISEAJOIDIOEEMNPOLGAANIOSTPT",
    "ALTANSGHIHRIESSUIONEINKIEGEMYMPTRELTOUBMEFLRHUNBUISOABTDWSSODEEAFDIAIULUDEANSEGYYANPOPREOWOCAOEARVARNHRSLELORWRYTDYTNFFKTYWUTAAO"
]

# For simplicity - all texts are converted to low cased.
PERMUTATIONS = [s.lower() for s in PERMUTATIONS]
PERMUTATION_LENGTH = 128

SCORE_BEAM_SIZE = 50
MAX_HYPOTHESES = 200
FINAL_TOP_RESULTS = 10

def search(lm_file: str, out_file: str):
    # Read the language model, and convert the permuted texts into word IDs.
    print('Search permutations - Start')
    print("\tRead LM")
    lang_model = ngram_lang_model.read_arpabo(lm_file)
    vocabulary = lang_model.vocabulary()
    permutations_ids = []

    print("\tConvert permutated texts to indexes")
    for i in range(len(PERMUTATIONS)):
        permutation_ids = []
        for letter in PERMUTATIONS[i]:
            permutation_ids.append(vocabulary.id(letter))
        permutations_ids.append(permutation_ids)

    # Create the base hypothesis.
    base_hypot = Hypothesis()
    for i in range(len(permutations_ids)):
        base_hypot.ordered_ids.append([Vocabulary.START_ID])

	# Expand permutation maps while killing
    # unreasonable paths, for keeping the computational
    # cost reasonable.
    print("\tSearch the permutation map in a viterbi-like procedure - Start")
    source_hypotheses = [base_hypot]
    for source_index in tqdm(range(PERMUTATION_LENGTH)):
        #print("\t\tExtend layer - #source items = " + str(len(source_hypotheses)))
        target_hypotheses = []

        for source_hypot in source_hypotheses:
            for target_ind in range(PERMUTATION_LENGTH):
                if target_ind in set(source_hypot.permutation):
                    # Skip target index that has already been
                    # mapped to during the source hypotheses prefix
                    # path.
                    continue

                # Extend the source hypothesis into the target index.
                target_hypot = source_hypot.extend(target_ind, permutations_ids, lang_model)
                target_hypotheses.append(target_hypot)

        #print("\t\tSource Index = " + str(source_index) + "#Raw target hypothesis = " + str(len(target_hypotheses)))

        sorted_hypotheses = sorted(target_hypotheses, key=lambda hypot: -(hypot.score))
        score_thresh = sorted_hypotheses[0].score - SCORE_BEAM_SIZE
        #print("\t\tscore threshold = {:.3f}".format(score_thresh))

        source_hypotheses.clear()
        for hypot in sorted_hypotheses:
            #print("\t\t\t#filtered hypotheses = " + str(len(source_hypotheses)) + ", score = {:.3f}".format(hypot.score))
            if len(source_hypotheses) >= MAX_HYPOTHESES or hypot.score < score_thresh:
                break

            source_hypotheses.append(hypot)

        #print("\t\tSource Index = " + str(source_index) + "#Filtered hypothesis = " + str(len(source_hypotheses)))
        target_hypotheses.clear()
    print("\tSearch the permutation map in a viterbi-like procedure - End")

	# Perform the final scoring of the top hypotheses.
    target_hypotheses = []
    for source_hypot in source_hypotheses:
        target_hypot = source_hypot.extend_final(lang_model)
        target_hypotheses.append(target_hypot)

    num_top_hypotheses = min(FINAL_TOP_RESULTS, len(target_hypotheses))
    final_hypotheses = sorted(target_hypotheses, key=lambda hypot: -(hypot.score))[0: num_top_hypotheses]

    # Print the top hypotheses.
    with open(out_file, "w", encoding='utf-8') as f:
        for rank, hypot in enumerate(final_hypotheses):
            print("Rank #" + str(rank) + ", lm={:.3f}".format(hypot.score) + ":")
            f.write("Rank #" + str(rank) + ", lm={:.3f}".format(hypot.score) + ":\n")
            for i in range(len(PERMUTATIONS)):
                print("in:  " + PERMUTATIONS[i])             
                f.write("in:  " + PERMUTATIONS[i] + "\n")
                out_str = "".join([PERMUTATIONS[i][hypot.permutation[j]] for j in range(PERMUTATION_LENGTH)])
                print("out: " + out_str)
                f.write("out: " + out_str + "\n")
                f.flush()
        print('Search permutations - End')
