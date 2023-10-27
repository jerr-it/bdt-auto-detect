import unittest
from src.stats.language import Language, G


class LanguageTest(unittest.TestCase):
    def test_g(self):
        self.assertEqual(G.convert("AAA123"), "[C;3][D;3]")
        self.assertEqual(G.convert("AAA123BBB"), "[C;3][D;3][C;3]")
        self.assertEqual(G.convert("AAA123BBBaaa"), "[C;3][D;3][C;3][c;3]")
        self.assertEqual(G.convert("..aaAA123BBBaaa"), "[.;2][c;2][C;2][D;3][C;3][c;3]")
        self.assertEqual(G.convert("!?-+#/@€.-"), "[!;1][?;1][-;1][+;1][#;1][/;1][@;1][€;1][.;1][-;1]")
