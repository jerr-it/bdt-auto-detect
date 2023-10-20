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

        return result


G = Language({
    str.isdigit: "D",
    str.isupper: "U",
    str.islower: "L",
    lambda x: True: "S",
})
