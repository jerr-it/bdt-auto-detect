import pandas as pd
import unittest

from src.stats.npmi import ValueColumnList


class TestNPMI(unittest.TestCase):
    def test_compatibility(self):
        # Example usage with either a DataFrame or a list of columns
        data1 = {'col1': [1, 2, 1, 2, 1],
                 'col2': [2, 2, 1, 2, 2]}
        df = pd.DataFrame(data1)

        data2 = [[1, 2, 1, 2, 1],
                 [2, 2, 1, 2, 2]]
        value_column_list_df = ValueColumnList(df)
        value_column_list_list = ValueColumnList(data2)

        # Should be compatible
        # print(value_column_list_df.compatible(1, 2, -0.5))
        self.assertTrue(value_column_list_df.compatible(1, 2, -0.5))

        # Should be incompatible
        # print(value_column_list_list.compatible(1, 3, -0.5))
        self.assertFalse(value_column_list_list.compatible(1, 3, -0.5))