import unittest
from src.stats.language import Language, G


class LanguageTest(unittest.TestCase):
    def test_g(self):
        self.assertEqual(G.convert("AAA123"), "UUUDDD")
        self.assertEqual(G.convert("AAA123BBB"), "UUUDDDUUU")
        self.assertEqual(G.convert("AAA123BBBaaa"), "UUUDDDUUULLL")
        self.assertEqual(G.convert("..aaAA123BBBaaa"), "SSLLUUDDDUUULLL")
        self.assertEqual(G.convert("!?-+#/@€.-"), "SSSSSSSSSS")
