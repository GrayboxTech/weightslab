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
        try:
            del got
        except Exception:
            pass
        gc.collect()
        # listing may not contain 'w' anymore
        names = GLOBAL_LEDGER.list_models()
        self.assertNotIn("w", names)

    def test_optimizer_live_update_through_proxy(self):
        # create a proxy placeholder by requesting before registration
        proxy = GLOBAL_LEDGER.get_optimizer('opt_live')
        # define a simple optimizer-like object
        class DummyOpt:
            def __init__(self, lr):
                self.lr = lr

        opt1 = DummyOpt(lr=0.1)
        # register first optimizer; since a proxy existed it should be updated in-place
        GLOBAL_LEDGER.register_optimizer('opt_live', opt1)

        handle = GLOBAL_LEDGER.get_optimizer('opt_live')
        # handle should reflect underlying object's attribute
        self.assertEqual(handle.lr, 0.1)

        # modify the optimizer in-place elsewhere and verify ledger reflects change
        opt1.lr = 0.2
        self.assertEqual(handle.lr, 0.2)

        # now register a new optimizer object under same name; proxy should update to new object
        opt2 = DummyOpt(lr=0.5)
        GLOBAL_LEDGER.register_optimizer('opt_live', opt2)
        # handle (proxy) should now forward to the new optimizer
        self.assertEqual(handle.lr, 0.5)


if __name__ == "__main__":
    unittest.main()
