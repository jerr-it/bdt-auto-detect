import itertools
from typing import Callable


class Language:
    """
    This class represents a Language
    The pattern map defines the mapping of a character to a pattern.
    See G() for an example.
    """
    def __init__(self, pattern_map: dict[Callable, Callable], threshold=0.0):
        self.pattern_map = pattern_map
        self.threshold = threshold
        self.h_minus = set()
        self.h_plus = set()

    def __hash__(self):
        hash((self.pattern_map, self.threshold))

    def rle(self, value: str) -> str:
        """
        Performs run-length encoding
        """
        result = ""
        current = ""
        count = 0
        for char in value:
            if char == current:
                count += 1
            else:
                if count > 0:
                    result += f"[{current};{count}]"
                current = char
                count = 1

        if count > 0:
            result += f"[{current};{count}]"

        return result

    def convert(self, value: str) -> str:
        result = ""

        for char in value:
            for pattern, generalization in self.pattern_map.items():
                if pattern(char):
                    result += generalization(char)
                    break

        return self.rle(result)


# H is the set of possible languages
H = {
    str.isupper: [lambda x: "C", lambda x: "L", lambda x: "A", lambda x: x],
    str.islower: [lambda x: "c", lambda x: "L", lambda x: "A", lambda x: x],
    str.isdigit: [lambda x: "D", lambda x: "A", lambda x: x],
    lambda x: True: [lambda x: "S", lambda x: "A", lambda x: x],
}

# L is the set of candidate languages
L: list[Language] = []
for candidate in itertools.product(*H.values()):
    L.append(Language(dict(zip(H.keys(), candidate))))

# G, crude generalization language
# Not to be confused with the G of the greedy algorithm in the paper
G = Language({
    str.isdigit: lambda x: "D",
    str.isupper: lambda x: "C",
    str.islower: lambda x: "c",
    lambda x: True: lambda x: x,
}, threshold=-0.3)
