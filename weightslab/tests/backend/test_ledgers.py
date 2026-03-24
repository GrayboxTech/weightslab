import gc
import unittest

from weightslab.backend.ledgers import GLOBAL_LEDGER, DEFAULT_NAME, Ledger, Proxy


class Dummy:
    def __init__(self, name):
        self.name = name


class LedgerTests(unittest.TestCase):
    def setUp(self):
        GLOBAL_LEDGER.clear()

    def test_empty_ledger_initialization(self):
        """Test that a fresh ledger is empty."""
        fresh_ledger = Ledger()
        self.assertEqual(fresh_ledger.list_models(), [])
        self.assertEqual(fresh_ledger.list_dataloaders(), [])
        self.assertEqual(fresh_ledger.list_optimizers(), [])
        self.assertEqual(fresh_ledger.list_hyperparams(), [])
        self.assertEqual(fresh_ledger.list_loggers(), [])
        self.assertEqual(fresh_ledger.list_signals(), [])
        self.assertEqual(fresh_ledger.list_checkpoint_managers(), [])
        self.assertEqual(fresh_ledger.list_dataframes(), [])

    def test_default_name_usage(self):
        """Test that DEFAULT_NAME ('main') is used when name is not provided."""
        d = Dummy("a")
        # Register without providing name - should use DEFAULT_NAME
        GLOBAL_LEDGER.register_model(model=d)
        self.assertIn(DEFAULT_NAME, GLOBAL_LEDGER.list_models())
        got = GLOBAL_LEDGER.get_model()  # Should get 'main' by default
        self.assertIs(got, d)

    def test_proxy_initialization_pattern(self):
        """Test that get before register returns Proxy(None), then updates on register."""
        # Get before register - should return Proxy(None)
        hp = GLOBAL_LEDGER.get_hyperparams()  # Uses DEFAULT_NAME

        # Proxy should exist but not have underlying object yet
        self.assertEqual(hp.get(), {})

        # Now register hyperparams
        params = {'learning_rate': 0.001, 'batch_size': 32}
        GLOBAL_LEDGER.register_hyperparams(params=params)

        # Same handle should now forward to registered object
        self.assertEqual(hp.get(), params)
        self.assertEqual(hp['learning_rate'], 0.001)
        self.assertEqual(hp['batch_size'], 32)

    def test_dataloaders_dict(self):
        """Test that get_dataloaders_dict returns dict of Proxy(None) for each name."""
        class DummyLoader:
            def __init__(self, name):
                self.name = name

        # Get dataloaders before registering - should return dict of Proxy(None)
        loaders = GLOBAL_LEDGER.get_dataloaders_dict(['train', 'val'])

        self.assertIn('train', loaders)
        self.assertIn('val', loaders)
        self.assertEqual(loaders['train'].get(), None)
        self.assertEqual(loaders['val'].get(), None)

        # Register actual loaders
        train_loader = DummyLoader('train')
        val_loader = DummyLoader('val')
        GLOBAL_LEDGER.register_dataloaders_dict({'train': train_loader, 'val': val_loader})

        # Same proxies should now forward to registered loaders
        self.assertEqual(loaders['train'].name, 'train')
        self.assertEqual(loaders['val'].name, 'val')

    def test_register_and_get_strong(self):
        d = Dummy("a")
        GLOBAL_LEDGER.register_model(d, name="m")
        self.assertIn("m", GLOBAL_LEDGER.list_models())
        got = GLOBAL_LEDGER.get_model("m")
        self.assertIs(got, d)

    def test_unregister(self):
        d = Dummy("a")
        GLOBAL_LEDGER.register_model(d, name="m")
        GLOBAL_LEDGER.unregister_model("m")
        self.assertNotIn("m", GLOBAL_LEDGER.list_models())

    def test_weak_registration(self):
        d = Dummy("weak")
        GLOBAL_LEDGER.register_model(d, name="w", weak=True)
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
        GLOBAL_LEDGER.get_optimizer('opt_live')  # Init opt with a proxy entry

        # define a simple optimizer-like object
        class DummyOpt:
            def __init__(self, lr):
                self.lr = lr

        opt1 = DummyOpt(lr=0.1)
        # register first optimizer; since a proxy existed it should be updated in-place
        GLOBAL_LEDGER.register_optimizer(opt1, name='opt_live')

        handle = GLOBAL_LEDGER.get_optimizer('opt_live')
        # handle should reflect underlying object's attribute
        self.assertEqual(handle.lr, 0.1)

        # modify the optimizer in-place elsewhere and verify ledger reflects change
        opt1.lr = 0.2
        self.assertEqual(handle.lr, 0.2)

        # now register a new optimizer object under same name; proxy should update to new object
        opt2 = DummyOpt(lr=0.5)
        GLOBAL_LEDGER.register_optimizer(opt2, name='opt_live')
        # handle (proxy) should now forward to the new optimizer
        self.assertEqual(handle.lr, 0.5)


    # ===== Comprehensive tests for all object types =====
    def test_model_default_name_proxy_pattern(self):
        """Test model registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        model_handle = GLOBAL_LEDGER.get_model()
        self.assertIsInstance(model_handle, Proxy)
        self.assertIsNone(model_handle.get())

        # Register without providing name - should use DEFAULT_NAME
        model_obj = Dummy("my_model")
        GLOBAL_LEDGER.register_model(model=model_obj)

        # Same handle should now reference the registered object
        self.assertIs(model_handle.get(), model_obj)
        self.assertEqual(GLOBAL_LEDGER.list_models(), [DEFAULT_NAME])

    def test_optimizer_default_name_proxy_pattern(self):
        """Test optimizer registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        opt_handle = GLOBAL_LEDGER.get_optimizer()
        self.assertIsInstance(opt_handle, Proxy)
        self.assertIsNone(opt_handle.get())

        # Register without providing name - should use DEFAULT_NAME
        opt_obj = Dummy("my_optimizer")
        GLOBAL_LEDGER.register_optimizer(optimizer=opt_obj)

        # Same handle should now reference the registered object
        self.assertIs(opt_handle.get(), opt_obj)
        self.assertEqual(GLOBAL_LEDGER.list_optimizers(), [DEFAULT_NAME])

    def test_hyperparams_default_name_proxy_pattern(self):
        """Test hyperparams registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        self.assertIsInstance(hp_handle, Proxy)
        self.assertEqual(hp_handle.get(), {})

        # Register without providing name - should use DEFAULT_NAME
        params = {'learning_rate': 0.01, 'batch_size': 64}
        GLOBAL_LEDGER.register_hyperparams(params=params)

        # Same handle should now reference the registered object
        self.assertEqual(hp_handle.get(), params)
        self.assertEqual(hp_handle['learning_rate'], 0.01)
        self.assertEqual(GLOBAL_LEDGER.list_hyperparams(), [DEFAULT_NAME])

    def test_proxy_get_key_default_mode_returns_live_proxy(self):
        """Default key get returns a live ValueProxy for dict-backed targets."""
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.register_hyperparams(params={"data_root": "C:/data/v1"})

        data_root = hp_handle.get("data_root")
        self.assertEqual(data_root, "C:/data/v1")
        self.assertTrue(hasattr(data_root, "set"))

        # Updating the underlying mapping is reflected through the live proxy.
        hp_handle["data_root"] = "C:/data/v2"
        self.assertEqual(data_root.get(), "C:/data/v2")

    def test_proxy_get_key_explicit_plain_value_mode(self):
        """proxy=False returns a plain snapshot value."""
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.register_hyperparams(params={"data_root": "C:/data/v1"})

        data_root = hp_handle.get("data_root", proxy=False)
        self.assertEqual(data_root, "C:/data/v1")

        hp_handle["data_root"] = "C:/data/v2"
        self.assertEqual(data_root, "C:/data/v1")

    def test_value_proxy_numeric_comparisons(self):
        """ValueProxy supports all standard numeric and string comparison operators."""
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.register_hyperparams(params={"lr": 0.01, "batch_size": 32, "tag": "v1"})

        lr = hp_handle.get("lr")
        bs = hp_handle.get("batch_size")
        tag = hp_handle.get("tag")

        # Equality
        self.assertTrue(lr == 0.01)
        self.assertTrue(0.01 == lr)
        self.assertFalse(lr != 0.01)

        # Ordering — ValueProxy on left
        self.assertTrue(lr < 1.0)
        self.assertTrue(lr <= 0.01)
        self.assertTrue(lr > 0.001)
        self.assertTrue(lr >= 0.01)

        # Ordering — plain value on left (uses __gt__/__ge__ via reflected ops)
        self.assertTrue(1.0 > lr)
        self.assertTrue(0.01 >= lr)
        self.assertTrue(0.001 < lr)

        # Integer / index coercion
        self.assertEqual(int(bs), 32)
        self.assertEqual(float(lr), 0.01)

        # Arithmetic
        self.assertAlmostEqual(lr + 0.09, 0.1)
        self.assertAlmostEqual(0.09 + lr, 0.1)
        self.assertEqual(bs * 2, 64)
        self.assertEqual(2 * bs, 64)
        self.assertEqual(bs - 2, 30)
        self.assertEqual(40 - bs, 8)
        self.assertEqual(bs // 3, 10)
        self.assertEqual(100 // bs, 3)
        self.assertEqual(bs % 5, 2)
        self.assertEqual(101 % bs, 5)
        self.assertAlmostEqual(bs / 2, 16.0)
        self.assertAlmostEqual(64 / bs, 2.0)

        # String comparison
        self.assertTrue(tag == "v1")
        self.assertTrue(tag < "v2")
        self.assertTrue("v0" < tag)

        # Hashability (usable in sets/dicts)
        s = {lr, 0.01}
        self.assertEqual(len(s), 1)

    def test_proxy_get_key_proxy_mode_live_read_and_write(self):
        """proxy=True returns a live key proxy that tracks and updates parent mapping."""
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.register_hyperparams(params={"data_root": "C:/data/v1"})

        data_root_proxy = hp_handle.get("data_root", proxy=True)
        self.assertEqual(data_root_proxy.get(), "C:/data/v1")

        # External mapping update is visible from the key proxy.
        hp_handle["data_root"] = "C:/data/v2"
        self.assertEqual(data_root_proxy.get(), "C:/data/v2")

        # Key proxy update writes back into parent mapping.
        data_root_proxy.set("C:/data/v3")
        self.assertEqual(hp_handle.get("data_root"), "C:/data/v3")

    def test_proxy_get_key_proxy_mode_survives_parent_replacement(self):
        """Live key proxy resolves against current parent object after re-registration."""
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.register_hyperparams(params={"data_root": "C:/data/v1"})

        data_root_proxy = hp_handle.get("data_root", proxy=True)
        self.assertEqual(data_root_proxy.get(), "C:/data/v1")

        # Re-register hyperparams under same name; existing parent proxy is updated.
        GLOBAL_LEDGER.register_hyperparams(params={"data_root": "C:/data/v4", "lr": 0.01})
        self.assertEqual(data_root_proxy.get(), "C:/data/v4")

        # Updates through key proxy should target the latest registered mapping.
        data_root_proxy.set("C:/data/v5")
        self.assertEqual(hp_handle.get("data_root"), "C:/data/v5")

    def test_proxy_get_key_proxy_mode_default_and_late_set(self):
        """Missing-key live proxy returns default and can later populate the key."""
        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.register_hyperparams(params={})

        data_root_proxy = hp_handle.get("data_root", default="C:/fallback", proxy=True)
        self.assertEqual(data_root_proxy.get(), "C:/fallback")

        data_root_proxy.set("C:/data/live")
        self.assertEqual(hp_handle.get("data_root"), "C:/data/live")

    def test_watch_or_edit_hyperparams_rebinds_caller_variable(self):
        """wl.watch_or_edit(parameters, flag='hyperparameters') rebinds the caller's
        local variable to the Proxy even without capturing the return value."""
        import weightslab as wl
        from weightslab.backend.ledgers import Proxy

        params = {"lr": 0.001, "batch_size": 32}
        wl.watch_or_edit(params, flag="hyperparameters", defaults=params)
        # After the call without assignment, `params` should now be a Proxy.
        self.assertIsInstance(params, Proxy)
        self.assertEqual(params["lr"], 0.001)

    def test_register_hyperparams_updates_existing_dict_in_place(self):
        """Re-registering hyperparams preserves dict identity for in-place workflows."""
        params = {"learning_rate": 0.001, "batch_size": 32}
        GLOBAL_LEDGER.register_hyperparams(params=params)

        hp_handle = GLOBAL_LEDGER.get_hyperparams()
        initial_id = id(hp_handle)

        GLOBAL_LEDGER.register_hyperparams(params={"learning_rate": 0.01, "momentum": 0.9})

        self.assertEqual(id(hp_handle), initial_id)
        self.assertEqual(hp_handle["learning_rate"], 0.01)
        self.assertEqual(hp_handle["momentum"], 0.9)
        self.assertNotIn("batch_size", hp_handle)

    def test_dataloader_default_name_proxy_pattern(self):
        """Test dataloader registration with default name and Proxy(None) pattern."""
        # Train
        # # Get before register - should create Proxy(None)
        loader_handle = GLOBAL_LEDGER.get_dataloader('train_loader')
        self.assertIsInstance(loader_handle, Proxy)
        self.assertIsNone(loader_handle.get())
        # # Register without providing name - should use DEFAULT_NAME
        train_loader_obj = Dummy("train_loader")
        GLOBAL_LEDGER.register_dataloader(dataloader=train_loader_obj, name=train_loader_obj.name)
        # # Same handle should now reference the registered object
        self.assertIs(loader_handle.get(), train_loader_obj)

        # Test
        # # Get before register - should create Proxy(None)
        loader_handle = GLOBAL_LEDGER.get_dataloader('test_loader')
        self.assertIsInstance(loader_handle, Proxy)
        self.assertIsNone(loader_handle.get())
        # # Register without providing name - should use DEFAULT_NAME
        train_loader_obj = Dummy("test_loader")
        GLOBAL_LEDGER.register_dataloader(dataloader=train_loader_obj, name=train_loader_obj.name)
        # # Same handle should now reference the registered object
        self.assertIs(loader_handle.get(), train_loader_obj)

        # Global verification
        self.assertEqual(GLOBAL_LEDGER.list_dataloaders(), ['train_loader', 'test_loader'])

    def test_logger_default_name_proxy_pattern(self):
        """Test logger registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        logger_handle = GLOBAL_LEDGER.get_logger()
        self.assertIsInstance(logger_handle, Proxy)
        self.assertIsNone(logger_handle.get())

        # Register without providing name - should use DEFAULT_NAME
        logger_obj = Dummy("my_logger")
        GLOBAL_LEDGER.register_logger(logger=logger_obj)

        # Same handle should now reference the registered object
        self.assertIs(logger_handle.get(), logger_obj)
        self.assertEqual(GLOBAL_LEDGER.list_loggers(), [DEFAULT_NAME])

    def test_signal_default_name_proxy_pattern(self):
        """Test signal registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        signal_handle = GLOBAL_LEDGER.get_signal()
        self.assertIsInstance(signal_handle, Proxy)
        self.assertIsNone(signal_handle.get())

        # Register without providing name - should use DEFAULT_NAME
        signal_obj = Dummy("my_signal")
        GLOBAL_LEDGER.register_signal(signal=signal_obj)

        # Same handle should now reference the registered object
        self.assertIs(signal_handle.get(), signal_obj)
        self.assertEqual(GLOBAL_LEDGER.list_signals(), [DEFAULT_NAME])

    def test_checkpoint_manager_default_name_proxy_pattern(self):
        """Test checkpoint manager registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        cm_handle = GLOBAL_LEDGER.get_checkpoint_manager()
        self.assertIsInstance(cm_handle, Proxy)
        self.assertIsNone(cm_handle.get())

        # Register without providing name - should use DEFAULT_NAME
        cm_obj = Dummy("my_checkpoint_manager")
        GLOBAL_LEDGER.register_checkpoint_manager(manager=cm_obj)

        # Same handle should now reference the registered object
        self.assertIs(cm_handle.get(), cm_obj)
        self.assertEqual(GLOBAL_LEDGER.list_checkpoint_managers(), [DEFAULT_NAME])

    def test_dataframe_default_name_proxy_pattern(self):
        """Test dataframe registration with default name and Proxy(None) pattern."""
        # Get before register - should create Proxy(None)
        df_handle = GLOBAL_LEDGER.get_dataframe()
        self.assertIsInstance(df_handle, Proxy)
        self.assertIsNone(df_handle.get())

        # Register without providing name - should use DEFAULT_NAME
        df_obj = Dummy("my_dataframe")
        GLOBAL_LEDGER.register_dataframe(dataframe=df_obj)

        # Same handle should now reference the registered object
        self.assertIs(df_handle.get(), df_obj)
        self.assertEqual(GLOBAL_LEDGER.list_dataframes(), [DEFAULT_NAME])

    def test_all_object_types_with_default_name(self):
        """Integration test: Register all object types with default names."""
        model = Dummy("model")
        optimizer = Dummy("optimizer")
        params = {'lr': 0.01}
        dataloader = Dummy("dataloader")
        logger = Dummy("logger")
        signal = Dummy("signal")
        checkpoint_manager = Dummy("checkpoint_manager")
        dataframe = Dummy("dataframe")

        # Register all with default names
        GLOBAL_LEDGER.register_model(model=model)
        GLOBAL_LEDGER.register_optimizer(optimizer=optimizer)
        GLOBAL_LEDGER.register_hyperparams(params=params)
        GLOBAL_LEDGER.register_dataloader(dataloader=dataloader)
        GLOBAL_LEDGER.register_logger(logger=logger)
        GLOBAL_LEDGER.register_signal(signal=signal)
        GLOBAL_LEDGER.register_checkpoint_manager(manager=checkpoint_manager)
        GLOBAL_LEDGER.register_dataframe(dataframe=dataframe)

        # Verify all are registered under DEFAULT_NAME
        self.assertEqual(GLOBAL_LEDGER.list_models(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_optimizers(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_hyperparams(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_dataloaders(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_loggers(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_signals(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_checkpoint_managers(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_dataframes(), [DEFAULT_NAME])

        # Verify all are retrievable without providing name
        self.assertIs(GLOBAL_LEDGER.get_model(), model)
        self.assertIs(GLOBAL_LEDGER.get_optimizer(), optimizer)
        self.assertEqual(GLOBAL_LEDGER.get_hyperparams(), params)
        self.assertIs(GLOBAL_LEDGER.get_dataloader(), dataloader)
        self.assertIs(GLOBAL_LEDGER.get_logger(), logger)
        self.assertIs(GLOBAL_LEDGER.get_signal(), signal)
        self.assertIs(GLOBAL_LEDGER.get_checkpoint_manager(), checkpoint_manager)
        self.assertIs(GLOBAL_LEDGER.get_dataframe(), dataframe)

    def test_get_before_register_empty_ledger(self):
        """Test that getting objects before registering creates proxies in ledger."""
        # Get all objects before registering anything
        GLOBAL_LEDGER.get_model()
        GLOBAL_LEDGER.get_optimizer()
        GLOBAL_LEDGER.get_hyperparams()
        GLOBAL_LEDGER.get_dataloader()
        GLOBAL_LEDGER.get_logger()
        GLOBAL_LEDGER.get_signal()
        GLOBAL_LEDGER.get_checkpoint_manager()
        GLOBAL_LEDGER.get_dataframe()

        # Ledger should now have entries for all (Proxy placeholders)
        self.assertEqual(GLOBAL_LEDGER.list_models(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_optimizers(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_hyperparams(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_dataloaders(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_loggers(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_signals(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_checkpoint_managers(), [DEFAULT_NAME])
        self.assertEqual(GLOBAL_LEDGER.list_dataframes(), [DEFAULT_NAME])

    def test_snapshot_after_all_registrations(self):
        """Test ledger snapshot with all object types registered."""
        # Register all types
        GLOBAL_LEDGER.register_model(model=Dummy("m"))
        GLOBAL_LEDGER.register_optimizer(optimizer=Dummy("o"))
        GLOBAL_LEDGER.register_hyperparams(params={'a': 1})
        GLOBAL_LEDGER.register_dataloader(dataloader=Dummy("d"))
        GLOBAL_LEDGER.register_logger(logger=Dummy("l"))
        GLOBAL_LEDGER.register_signal(signal=Dummy("s"))
        GLOBAL_LEDGER.register_checkpoint_manager(manager=Dummy("c"))
        GLOBAL_LEDGER.register_dataframe(dataframe=Dummy("df"))

        snapshot = GLOBAL_LEDGER.snapshot()

        # All should show DEFAULT_NAME in snapshot
        self.assertEqual(snapshot['models'], [DEFAULT_NAME])
        self.assertEqual(snapshot['optimizers'], [DEFAULT_NAME])
        self.assertEqual(snapshot['hyperparams'], [DEFAULT_NAME])
        self.assertEqual(snapshot['dataloaders'], [DEFAULT_NAME])
        self.assertEqual(snapshot['loggers'], [DEFAULT_NAME])
        self.assertEqual(snapshot['checkpoint_managers'], [DEFAULT_NAME])
        self.assertIn('checkpoint_managers', snapshot)



if __name__ == "__main__":
    unittest.main()
