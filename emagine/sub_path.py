import copy

class CorrectSubPath:
    def __init__(self, depth: int, already_mapped: dict) -> None:
        self.depth = depth
        self.already_mapped = already_mapped

class PendingSubPath:
    def __init__(self, depth: int, no_match_depth: int, already_mapped: dict, matches: list) -> None:
        self.total_depth = depth
        self.no_match_depth = no_match_depth
        self.already_mapped = copy.deepcopy(already_mapped)
        self.matches = matches
