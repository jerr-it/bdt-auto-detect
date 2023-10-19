"""
This module contains the necessary functions to generalize values into patterns and calculate the NPMI score of patterns.
"""
import pandas as pd


def convert_to_pattern(series: pd.Series):
    """
    Generalizes all entries in a series by converting them all into a pattern. The pattern is created by grouping the
    characters into classes (digits, upper and lower case letters) and leaving all other characters as they are.
    (Also known as G() in the paper)
    """



