import itertools
from typing import Callable


class Language:
    """
    This class represents a Language
    The pattern map defines the mapping of a character to a pattern.
    See G() for an example.

    The 'pattern_map' may not be changed so that the output of the hashing function remains consistent.
    """
    def __init__(self, pattern_map: dict[Callable, Callable], threshold=0.0):
        self.pattern_map = pattern_map
        self.threshold = threshold
        self.h_minus = set()
        self.h_plus = set()

    def __hash__(self):
        return hash((tuple(self.pattern_map.items()), self.threshold))

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

    @staticmethod
    def run_length_encode(data: str) -> str:
        return "".join(f"{x}{sum(1 for _ in y)}" for x, y in itertools.groupby(data))

    def convert(self, value: str) -> str:
        result = ""

        for char in value:
            for pattern, generalization in self.pattern_map.items():
                if pattern(char):
                    result += generalization(char)
                    break

        #return Language.run_length_encode(result)
        return result


def identity(x: str) -> str:
    return x


def C(x: str) -> str:
    return "C"


def c(x: str) -> str:
    return "c"


def A(x: str) -> str:
    return "A"


def D(x: str) -> str:
    return "D"


def LL(x: str) -> str:
    return "L"


def S(x: str) -> str:
    return "S"


def other(x: str) -> bool:
    return True


# H is the set of possible languages
H = {
    str.isupper: [C, LL, A, identity],
    str.islower: [c, LL, A, identity],
    str.isdigit: [D, A, identity],
    other: [S, A, identity],
}

# L is the set of candidate languages
L: list[Language] = []
for candidate in itertools.product(*H.values()):
    L.append(Language(dict(zip(H.keys(), candidate))))

# G, crude generalization language
# Not to be confused with the G of the greedy algorithm in the paper
G = Language({
    str.isdigit: D,
    str.isupper: C,
    str.islower: c,
    other: identity,
}, threshold=-0.3)
