from enum import Enum


class Label(str, Enum):
    """
    Represents a label, either + or -.
    """
    POSITIVE = "+"
    NEGATIVE = "-"