
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
            
            wrappered_fwd(
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

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_signals_classification(self, mock_gm, mock_get_dataframe):
        """Test save_signals with classification task type."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2, 3])
            signals = {"cls_loss": torch.tensor(0.5)}
            preds = torch.tensor([0.8, 0.2, 0.9])
            targets = torch.tensor([1, 0, 1])

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertIn("signals//cls_loss", call_kwargs['losses'])
            self.assertEqual(len(call_kwargs['sample_ids']), 3)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_signals_segmentation(self, mock_gm, mock_get_dataframe):
        """Test save_signals with segmentation task type (mask data)."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2])
            signals = {"seg_loss": torch.tensor(0.3)}
            # Segmentation: spatial masks
            preds = torch.randint(0, 5, (2, 256, 256), dtype=torch.uint8)
            targets = torch.randint(0, 5, (2, 256, 256), dtype=torch.uint8)

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertIn("signals//seg_loss", call_kwargs['losses'])
            self.assertEqual(len(call_kwargs['sample_ids']), 2)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_signals_detection(self, mock_gm, mock_get_dataframe):
        """Test save_signals with detection task type (bbox data)."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2])
            signals = {"det_loss": torch.tensor(0.4)}
            # Detection: list of bbox arrays (variable number of boxes per image)
            preds = [
                torch.tensor([[10.0, 20.0, 100.0, 150.0], [200.0, 250.0, 400.0, 450.0]]),
                torch.tensor([[15.0, 25.0, 110.0, 160.0]])
            ]
            targets = [
                torch.tensor([[12.0, 22.0, 102.0, 152.0], [202.0, 252.0, 402.0, 452.0]]),
                torch.tensor([[14.0, 24.0, 109.0, 159.0]])
            ]

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertIn("signals//det_loss", call_kwargs['losses'])
            self.assertEqual(len(call_kwargs['sample_ids']), 2)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_signals_upsert_behavior(self, mock_gm, mock_get_dataframe):
        """Test that save_signals correctly upserts signals into dataframe."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 5
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = [100, 101, 102]
            signals = {
                "loss/train": 0.25,
                "accuracy/train": 0.95,
                "f1_score/train": 0.92
            }

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                log=True
            )

            # Verify enqueue_batch was called (signals are queued for upsert)
            self.assertTrue(mock_df.enqueue_batch.called)

            # Check that all signals are present
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            losses = call_kwargs['losses']
            self.assertIn("signals//loss/train", losses)
            self.assertIn("signals//accuracy/train", losses)
            self.assertIn("signals//f1_score/train", losses)

            # Verify batch ids match
            self.assertEqual(len(call_kwargs['sample_ids']), 3)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_signals_batch_processing(self, mock_gm, mock_get_dataframe):
        """Test that save_signals correctly processes batches for all task types."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 10
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            # Classification case
            wl.save_signals(
                signals={"cls_loss": torch.tensor([0.1, 0.2, 0.3])},
                batch_ids=torch.tensor([1, 2, 3]),
                log=True
            )

            # Segmentation case
            wl.save_signals(
                signals={"seg_loss": torch.tensor(0.15)},
                batch_ids=torch.tensor([4, 5]),
                preds=torch.ones((2, 64, 64), dtype=torch.uint8),
                targets=torch.zeros((2, 64, 64), dtype=torch.uint8),
                log=True
            )

            # Detection case
            wl.save_signals(
                signals={"det_loss": torch.tensor(0.2)},
                batch_ids=torch.tensor([6, 7]),
                preds=torch.rand((2, 5, 4)),  # 5 boxes, 4 coords
                targets=torch.rand((2, 4, 4)),  # 4 boxes, 4 coords
                log=True
            )

            # Verify all were queued correctly
            self.assertEqual(mock_df.enqueue_batch.call_count, 3)

if __name__ == "__main__":
    unittest.main()
