import itertools
from typing import Callable


class Language:
    """
    This class represents a Language
    The pattern map defines the mapping of a character to a pattern.
    See G() for an example.
    """
    def __init__(self, pattern_map: dict[Callable, str]):
        self.pattern_map = pattern_map

    def convert(self, value: str) -> str:
        result = ""

        for char in value:
            for pattern in self.pattern_map:
                if pattern(char):
                    result += self.pattern_map[pattern]
                    break

        return result


# H is the set of possible languages
H = {
    str.isupper: ["Lu", "L", "A"],
    str.islower: ["Ll", "L", "A"],
    str.isdigit: ["D", "A"],
    lambda x: True: ["S", "A"],
}

# L is the set of candidate languages
L = []
for candidate in itertools.product(*H.values()):
    L.append(Language(dict(zip(H.keys(), candidate))))

# G, crude generalization language
# Not to be confused with the G of the greedy algorithm in the paper
G = Language({
    str.isdigit: "D",
    str.isupper: "U",
    str.islower: "L",
    lambda x: True: "S",
})
