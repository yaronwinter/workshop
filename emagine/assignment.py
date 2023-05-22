from emagine.sub_path import CorrectSubPath, PendingSubPath
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
MIN_VALID_DEPTH = 5
MAX_SEARCH_DEPTH = 10

PERMUTATIONS = [s.lower() for s in PERMUTATIONS]

def find_permutation_map(lexicon_file: str) -> dict:
    correct_sub_path = CorrectSubPath(depth=-1, already_mapped={})

    with open(lexicon_file, "r", encoding="utf-8") as f:
        lexicon = set([word.strip() for word in f.readlines()])

    while correct_sub_path.depth < (PERMUTATION_LENGTH - 1):
        curr_pending_path = PendingSubPath(depth=correct_sub_path.depth, no_match_depth=0, already_mapped=correct_sub_path.already_mapped, matches=[])
        pending_sub_path = construct_pending_path(prev_valid_depth=correct_sub_path.depth, prev_sub_path=curr_pending_path, lexicon=lexicon)
        if pending_sub_path is None:
            return None
        
        correct_sub_path.depth = pending_sub_path.total_depth
        correct_sub_path.already_mapped = copy.deepcopy(pending_sub_path.already_mapped)

    return correct_sub_path


def transform_text(from_depth: int, to_depth: int, permutation: str, depth_map: dict) -> str:
    out_text = ""
    for i in range(from_depth, to_depth):
        depth = from_depth + i
        out_text += permutation[depth_map[depth]]

    return out_text

def check_match_validity(from_depth: int, to_depth: int, lexicon: set, depths_map: dict) -> bool:
    for permutation in PERMUTATIONS:
        transformed_text = transform_text(from_depth, to_depth, permutation, depths_map)
        if transformed_text in lexicon:
            return True
    return False


def check_depth_validity(prev_matches: list, to_depth: int, lexicon: set, depths_map: dict) -> list:
    curr_matches = []
    for m in prev_matches:
        if check_match_validity(m[0], to_depth, lexicon, depths_map):
            curr_matches.append((m[0], to_depth,))
            return curr_matches
        
        curr_matches.append(m)
    return None



def construct_pending_path(prev_valid_depth: int, prev_sub_path: PendingSubPath, lexicon: set) -> PendingSubPath:
    if prev_sub_path.no_match_depth > MAX_SEARCH_DEPTH:
        return None
    
    already_mapped = set(prev_sub_path.already_mapped.values())
    for i in range(PERMUTATION_LENGTH):
        if i in already_mapped:
            continue

        curr_depth = prev_sub_path.total_depth + 1
        curr_map = copy.deepcopy(prev_sub_path.already_mapped)
        curr_map[curr_depth] = i
        prev_matches = copy.deepcopy(prev_sub_path.matches)
        prev_matches.append((curr_depth - prev_sub_path.no_match_depth, prev_sub_path.no_match_depth + 1,))

        new_matches = check_depth_validity(prev_matches, curr_depth+1, lexicon, curr_map)
        if new_matches is None:
            curr_pending_path = PendingSubPath(curr_depth, prev_sub_path.no_match_depth + 1, curr_map, prev_sub_path.matches)
        else:
            curr_pending_path = PendingSubPath(curr_depth, 0, curr_map, new_matches)
            if (curr_depth - prev_valid_depth) >= MIN_VALID_DEPTH:
                return curr_pending_path
            
        return construct_pending_path(prev_valid_depth, curr_pending_path, lexicon)
    
    return None
