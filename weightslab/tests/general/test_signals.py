
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import weightslab as wl
from weightslab.src import _REGISTERED_SIGNALS, signal, compute_signals, wrappered_fwd

class TestSignals(unittest.TestCase):
    def setUp(self):
        # Clear registered signals before each test
        _REGISTERED_SIGNALS.clear()
        
    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    def test_signal_registration(self):
        """Test that @wl.signal registers the function correctly."""
        @wl.signal(name="test_sig")
        def my_signal(x):
            return x
            
        self.assertIn("test_sig", _REGISTERED_SIGNALS)
        self.assertEqual(_REGISTERED_SIGNALS["test_sig"], my_signal)
        
    def test_signal_metadata(self):
        """Test that metadata is attached to the signal function."""
        @wl.signal(name="meta_sig", subscribe_to="loss", compute_every_n_steps=5)
        def my_signal(x):
            return x
            
        self.assertTrue(hasattr(my_signal, "_wl_signal_meta"))
        self.assertEqual(my_signal._wl_signal_meta["subscribe_to"], "loss")
        self.assertEqual(my_signal._wl_signal_meta["compute_every_n_steps"], 5)
        self.assertEqual(my_signal._wl_signal_name, "meta_sig")

    @patch("weightslab.src.get_dataframe")
    def test_compute_signals_static(self, mock_get_dataframe):
        """Test computing static signals on a dataset."""
        # Mock dataframe
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df
        # Force reload of DATAFRAME_M in src if needed, or patch it directly
        with patch("weightslab.src.DATAFRAME_M", mock_df):
            
            # Define signal
            @wl.signal(name="mean_val")
            def compute_mean(item):
                return np.mean(item)
                
            # Dummy dataset
            dataset = [np.array([1, 2, 3]), np.array([4, 5, 6])]
            
            # Run compute_signals
            compute_signals(dataset, origin="train")
            
            # Verify upsert was called
            self.assertTrue(mock_df.upsert_df.called)
            args, kwargs = mock_df.upsert_df.call_args
            df_arg = args[0]
            
            self.assertEqual(len(df_arg), 2)
            self.assertIn("signals_mean_val", df_arg.columns)
            self.assertEqual(df_arg.iloc[0]["signals_mean_val"], 2.0)
            self.assertEqual(df_arg.iloc[1]["signals_mean_val"], 5.0)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src.list_models")
    @patch("weightslab.src._gm") # get_model
    def test_dynamic_signal_subscription(self, mock_gm, mock_list_models, mock_get_dataframe):
        """Test dynamic signal execution via wrappered_fwd."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df
        
        # Mock model for step counter
        mock_model = MagicMock()
        mock_model.current_step = 10
        mock_gm.return_value = mock_model
        mock_list_models.return_value = ["main"]
        
        with patch("weightslab.src.DATAFRAME_M", mock_df):
            # 1. Define subscriber signal
            # It expects (sample_id, value, dataframe)
            mock_subscriber = MagicMock(return_value=0.5)
            
            @wl.signal(name="dynamic_sig", subscribe_to="source_metric")
            def dynamic_sig(sample_id, value, dataframe=None, origin="train"):
                return mock_subscriber(sample_id=sample_id, value=value, dataframe=dataframe)

            # 2. Simulate the 'source_metric' being executed via wrappered_fwd
            # validation logic in wrappered_fwd checks keys in _REGISTERED_SIGNALS
            
            original_forward = MagicMock(return_value=torch.tensor([1.0, 2.0]))
            
            # Inputs to forward
            args = (torch.zeros(2), torch.zeros(2)) # input, target
            kwargs = {
                "batch_ids": torch.tensor([100, 101]),
                "log": False,
                # signals arg is usually populated by wrapper, but here we simulate what wrapper does
                "signals": None 
            }
            
            # Call the wrapper logic directly? 
            # wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw)
            # kwargs passed to wrappered_fwd are the decorator kwargs (e.g. name="source_metric")
            decorator_kwargs = {"name": "source_metric"}
            
            # wrappered_fwd signature: (original_forward, kwargs, reg_name, *a, **kw)
            # wait, wrappered_fwd definition: def wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw):
            # 'kwargs' in definition corresponds to decorator kwargs.
            # '*a, **kw' are the runtime args.
            
            res = wrappered_fwd(
                original_forward, 
                decorator_kwargs, 
                "source_metric", 
                *args, 
                **kwargs
            )
            
            # 3. Verify subscriber was called
            # Should be called once per sample in batch (size 2)
            self.assertEqual(mock_subscriber.call_count, 2)
            
            # Check call args
            # Call 1: id=100, val=1.0 (mean of [1.0])
            call_args_list = mock_subscriber.call_args_list
            self.assertEqual(call_args_list[0][1]['sample_id'], 100)
            self.assertAlmostEqual(call_args_list[0][1]['value'], 1.0)
            
            # Call 2: id=101, val=2.0
            self.assertEqual(call_args_list[1][1]['sample_id'], 101)
            self.assertAlmostEqual(call_args_list[1][1]['value'], 2.0)
            
            # 4. Verify save_signals was called (implied by mock_df.enqueue_batch or upsert_df being called if implemented in save_signals)
            # save_signals calls DATAFRAME_M.enqueue_batch
            self.assertTrue(mock_df.enqueue_batch.called)
            
            # Check that "dynamic_sig" is in the losses passed to enqueue_batch
            call_args = mock_df.enqueue_batch.call_args
            losses_data = call_args[1]['losses']
            self.assertIn('signals//dynamic_sig', losses_data)
            
    def test_frequency_control(self):
        """Test compute_every_n_steps."""
        with patch("weightslab.src.list_models", return_value=["main"]), \
             patch("weightslab.src._gm") as mock_gm, \
             patch("weightslab.src.DATAFRAME_M", MagicMock()):
             
            mock_model = MagicMock()
            mock_model.current_step = 3 # Not divisible by 2
            mock_gm.return_value = mock_model
            
            mock_sub = MagicMock()
            @wl.signal(name="freq_sig", subscribe_to="src", compute_every_n_steps=2)
            def freq_sig(**kwargs):
                mock_sub()
                return 0

            # Run trigger
            original_forward = MagicMock(return_value=torch.tensor([1.0]))
            wrappered_fwd(
                original_forward, 
                {"name": "src"}, 
                "src", 
                torch.zeros(1), torch.zeros(1), 
                batch_ids=torch.tensor([1])
            )
            
            # Should NOT run because 3 % 2 != 0
            mock_sub.assert_not_called()
            
            # Change step to 4
            mock_model.current_step = 4
             # Run trigger
            wrappered_fwd(
                original_forward, 
                {"name": "src"}, 
                "src", 
                torch.zeros(1), torch.zeros(1), 
                batch_ids=torch.tensor([1])
            )
            
            # Should run
            mock_sub.assert_called()

if __name__ == "__main__":
    unittest.main()
