import copy

# An object that is used for the recursive search
class PendingSubPath:
    def __init__(self, depth: int, no_match_depth: int, already_mapped: dict, matches: list, converted_text: str) -> None:
        self.depth = depth
        self.no_match_depth = no_match_depth
        self.already_mapped = copy.deepcopy(already_mapped)
        self.matches = copy.deepcopy(matches)
        self.converted_text = converted_text
