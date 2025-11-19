import gc
import unittest

from weightslab.ledgers import GLOBAL_LEDGER


class Dummy:
    def __init__(self, name):
        self.name = name


class LedgerTests(unittest.TestCase):
    def setUp(self):
        GLOBAL_LEDGER.clear()

    def test_register_and_get_strong(self):
        d = Dummy("a")
        GLOBAL_LEDGER.register_model("m", d)
        self.assertIn("m", GLOBAL_LEDGER.list_models())
        got = GLOBAL_LEDGER.get_model("m")
        self.assertIs(got, d)

    def test_unregister(self):
        d = Dummy("a")
        GLOBAL_LEDGER.register_model("m", d)
        GLOBAL_LEDGER.unregister_model("m")
        self.assertNotIn("m", GLOBAL_LEDGER.list_models())

    def test_weak_registration(self):
        d = Dummy("weak")
        GLOBAL_LEDGER.register_model("w", d, weak=True)
        # object should be present while strong ref exists
        self.assertIn("w", GLOBAL_LEDGER.list_models())
        got = GLOBAL_LEDGER.get_model("w")
        self.assertIs(got, d)
        # drop strong refs and force GC; weakref should disappear
        del d
        gc.collect()
        # listing may not contain 'w' anymore
        names = GLOBAL_LEDGER.list_models()
        self.assertNotIn("w", names)


if __name__ == "__main__":
    unittest.main()
