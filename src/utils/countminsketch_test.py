import unittest

import dill

from src.utils.countminsketch import CMSketch


# class CMSTest(unittest.TestCase):
#     def test_pickle(self):
#         ...
#         #self.assertTrue(False)
#         #with open("test.pkl", "wb") as f:
#         #    dill.dump(cms, f)
#         #self.assertTrue(False)
#         #with open("test.pkl", "rb") as f:
#         #    self.assertTrue(False)
#         #    cms_reloaded = dill.load(f)
#         #    self.assertTrue(False)
#         #    self.assertEqual(cms_reloaded.get("testkey"), 1)
#         #    self.assertEqual(cms_reloaded.get("testkey2"), 3)


if __name__ == '__main__':
    cms = CMSketch("test_file_name.txt", 1000, 8)
    cms.inc("testkey", 1)
    cms.inc("testkey2", 3)

    with open("test.pkl", "wb") as f:
        dill.dump(cms, f)

    with open("test.pkl", "rb") as f:
        cms_reloaded = dill.load(f)
        print(str(cms_reloaded.get("testkey")) + ":" + str(1))
        print(str(cms_reloaded.get("testkey2")) + ":" + str(3))