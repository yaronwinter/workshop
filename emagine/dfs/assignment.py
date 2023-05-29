from emagine.sub_path import PendingSubPath
import copy

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

PERMUTATION_LENGTH = 128
MIN_VALID_DEPTH = 6
MAX_SEARCH_DEPTH = 6

# For simplicity - all texts are converted to low cased.
PERMUTATIONS = [s.lower() for s in PERMUTATIONS]
PIVOT = PERMUTATIONS[0]

def find_permutation_map(lexicon_file: str) -> dict:
    with open(lexicon_file, "r", encoding="utf-8") as f:
        lexicon = set([x.strip() for x in f.readlines()])

    log_file = open("debug.txt", "w", encoding="utf-8")
    log_file.write("Lex Size = " + str(len(lexicon)) + "\n\n")
    root_sub_path = PendingSubPath(depth=-1, no_match_depth=0, already_mapped={}, matches=[], converted_text="")
    permutation_map = search_permutation(root_sub_path, lexicon, log_file)
    log_file.close()
    return permutation_map


# Use the temporal (prefix) permutation for convert a text
# section to its possibly original content.
def transform_text(from_depth: int, to_depth: int, permutation: str, permutation_map: dict) -> str:
    out_text = ""
    for i in range(from_depth, to_depth):
        depth = i
        out_text += permutation[permutation_map[depth]]
    return out_text


# Check whether the current section can be converted by the temporal (prefix) permutaion
# into a lexical word.
# Here we assume that although the strings are stemmed by onscure source, one of them should
# have a correct word in this tested section.
# Another assumtion is that long enough section (e.g. > 5) is not likely to be converted
# into lexical word, unless the permutation we found for this section is correct.
def check_match_validity(from_depth: int, to_depth: int, lexicon: set, permutation_map: dict, debug_file) -> bool:
    for permutation in PERMUTATIONS[:1]:
        #debug_file.write(f"\tcheck_match_validity({from_depth}, {to_depth})\n")
        #debug_file.write("\t\tpermutation: " + permutation + "\n")
        transformed_text = transform_text(from_depth, to_depth, permutation, permutation_map)
        #debug_file.write("\t\ttransformed: " + transformed_text + ", " + str(permutation_map) + ", " + str(from_depth) + ", " + str(to_depth) + "\n")
        if transformed_text in lexicon:
            #debug_file.write("\t\t\tIs Valid!!" + transformed_text + "\n")
            #debug_file.flush()
            return transformed_text
    #debug_file.write("\t\t\tNot Valid?!\n")
    #debug_file.flush()
    return ""


# The recursive implementaion of the DFS
def search_permutation(father_pending_path: PendingSubPath, lexicon: set, debug_file) -> PendingSubPath:
    if father_pending_path.no_match_depth > MAX_SEARCH_DEPTH:
        return None
    
    if father_pending_path.depth >= PERMUTATION_LENGTH:
        return father_pending_path
    
    depth = str(father_pending_path.depth)
    matches = str(father_pending_path.matches)
    mapped = str(father_pending_path.already_mapped)
    transformed = ("" if father_pending_path.depth < 0 else transform_text(0, father_pending_path.depth+1, PERMUTATIONS[0], father_pending_path.already_mapped))
    print("search_permutation: depth=" + depth + ", matches=" + matches + ", map=" + mapped + ", tt=" + transformed)
    #debug_file.write("search_permutation: depth=" + str(father_pending_path.depth) + ", matches=" + str(father_pending_path.matches) + "\n")
    
    brother = copy.deepcopy(father_pending_path)
    son_depth = father_pending_path.depth + 1
    for i in range(PERMUTATION_LENGTH):
        if i in father_pending_path.already_mapped:
            continue
        
        for j in range(PERMUTATION_LENGTH):
            if j in brother.already_mapped:
                continue

            son_mapped = copy.deepcopy(brother.already_mapped)
            son_mapped[son_depth] = j
            son_matches = copy.deepcopy(father_pending_path.matches)
            son_converted_text = brother.converted_text + PIVOT[j]
            from_depth = son_depth - father_pending_path.no_match_depth
            to_depth = son_depth + 1
            curr_word = son_converted_text[from_depth:to_depth]
            new_match = (curr_word if curr_word in lexicon else "")

            #new_match = check_match_validity(from_depth, to_depth, lexicon, son_mapped, debug_file)

            if len(new_match) == 0:
                son_pending_path = PendingSubPath(son_depth, brother.no_match_depth + 1, son_mapped, brother.matches, son_converted_text)
            else:
                son_matches.append((from_depth, to_depth, new_match,))
                son_pending_path = PendingSubPath(son_depth, 0, son_mapped, son_matches, son_converted_text)

            son_pending_path = search_permutation(father_pending_path=son_pending_path, lexicon=lexicon, debug_file=debug_file)
            if son_pending_path is not None:
                return son_pending_path
            
        brother.already_mapped[brother.depth] = i
        brother.converted_text = brother.converted_text[:brother.depth] + PIVOT[i]

    return None
