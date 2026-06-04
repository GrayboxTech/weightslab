
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import weightslab as wl
from weightslab.src import _REGISTERED_SIGNALS, compute_signals, wrappered_fwd

class TestSignals(unittest.TestCase):
    def setUp(self):
        # Clear registered signals before each test
        _REGISTERED_SIGNALS.clear()

    def tearDown(self):
        _REGISTERED_SIGNALS.clear()

    # =========================================================================
    # Input Type Tests - Different signal and data input formats
    # =========================================================================

    def test_signals_with_numpy_inputs(self):
        """Test save_signals with numpy array inputs for predictions and targets."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                batch_ids = np.array([10, 11, 12])
                signals = {"loss": np.array(0.5)}
                preds = np.array([0.9, 0.1, 0.8])
                targets = np.array([1, 0, 1])

                wl.save_signals(
                    signals=signals,
                    batch_ids=batch_ids,
                    preds=preds,
                    targets=targets,
                    log=True
                )

                self.assertTrue(mock_df.enqueue_batch.called)

    def test_signals_with_list_batch_ids(self):
        """Test save_signals with list batch IDs instead of tensors."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                batch_ids = [20, 21, 22]  # list instead of tensor
                signals = {"loss": 0.3}

                wl.save_signals(
                    signals=signals,
                    batch_ids=batch_ids,
                    log=True
                )

                self.assertTrue(mock_df.enqueue_batch.called)
                call_kwargs = mock_df.enqueue_batch.call_args[1]
                self.assertEqual(len(call_kwargs['sample_ids']), 3)

    def test_signals_with_scalar_values(self):
        """Test save_signals with scalar signal values."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                batch_ids = torch.tensor([30, 31])
                signals = {
                    "loss": 0.25,  # scalar float
                    "accuracy": 0.95,
                    "f1": np.float32(0.92)
                }

                wl.save_signals(
                    signals=signals,
                    batch_ids=batch_ids,
                    log=True
                )

                self.assertTrue(mock_df.enqueue_batch.called)
                call_kwargs = mock_df.enqueue_batch.call_args[1]
                self.assertIn("signals//loss", call_kwargs['losses'])

    def test_signals_with_multidimensional_tensors(self):
        """Test save_signals with multidimensional tensor signals."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                batch_ids = torch.tensor([40, 41, 42])
                # Multidimensional signal (e.g., per-layer losses)
                signals = {"layer_losses": torch.tensor([[0.1, 0.2], [0.15, 0.25], [0.12, 0.22]])}

                wl.save_signals(
                    signals=signals,
                    batch_ids=batch_ids,
                    log=True
                )

                self.assertTrue(mock_df.enqueue_batch.called)

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

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_signals_with_empty_batch_ids(self):
        """Test save_signals with empty batch IDs."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                batch_ids = torch.tensor([])
                signals = {"loss": torch.tensor(0.5)}

                # Should not crash, but should skip logging
                wl.save_signals(
                    signals=signals,
                    batch_ids=batch_ids,
                    log=True
                )

    def test_signals_with_none_batch_ids(self):
        """Test save_signals with None batch IDs."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                signals = {"loss": torch.tensor(0.5)}

                # Should not crash
                wl.save_signals(
                    signals=signals,
                    batch_ids=None,
                    log=False  # Don't log without IDs
                )

    def test_signal_with_mixed_data_types(self):
        """Test signal registration and computation with mixed data types."""
        @wl.signal(name="mixed_type_signal")
        def compute_mixed(item, **kwargs):
            if isinstance(item, (list, tuple)):
                return np.mean(item)
            elif isinstance(item, torch.Tensor):
                return item.mean().item()
            elif isinstance(item, np.ndarray):
                return np.mean(item)
            return float(item)

        self.assertIn("mixed_type_signal", _REGISTERED_SIGNALS)

        # Test with different types
        signal_fn = _REGISTERED_SIGNALS["mixed_type_signal"]

        result_list = signal_fn([1, 2, 3])
        self.assertAlmostEqual(result_list, 2.0)

        result_tensor = signal_fn(torch.tensor([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result_tensor, 2.0)

        result_numpy = signal_fn(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result_numpy, 2.0)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_save_signals_with_dict_predictions(self, mock_gm, mock_get_dataframe):
        """Test save_signals with dict-structured predictions (e.g., multi-output model)."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2])
            signals = {"loss": torch.tensor(0.5)}
            # Multi-output predictions (e.g., bounding boxes + class probabilities)
            preds = {
                "bboxes": [torch.tensor([[10.0, 20.0, 100.0, 150.0]]), torch.tensor([[15.0, 25.0, 110.0, 160.0]])],
                "classes": torch.tensor([[0.9, 0.1], [0.85, 0.15]])
            }

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)

    # =========================================================================
    # Signal Composition and Combination Tests
    # =========================================================================

    def test_multiple_signals_in_single_call(self):
        """Test saving multiple signals in a single save_signals call."""
        with patch("weightslab.src.get_dataframe") as mock_get_df, \
             patch("weightslab.src._gm") as mock_gm:

            mock_df = MagicMock()
            mock_get_df.return_value = mock_df
            mock_model = MagicMock()
            mock_model.current_step = 1
            mock_gm.return_value = mock_model

            with patch("weightslab.src.DATAFRAME_M", mock_df):
                batch_ids = torch.tensor([1, 2, 3])
                signals = {
                    "loss/cls": torch.tensor(0.3),
                    "loss/reg": torch.tensor(0.15),
                    "loss/total": torch.tensor(0.45),
                    "accuracy": torch.tensor(0.92),
                    "f1_score": torch.tensor(0.89)
                }

                wl.save_signals(
                    signals=signals,
                    batch_ids=batch_ids,
                    log=True
                )

                self.assertTrue(mock_df.enqueue_batch.called)
                call_kwargs = mock_df.enqueue_batch.call_args[1]
                losses = call_kwargs['losses']

                # Verify all signals are present
                self.assertIn("signals//loss/cls", losses)
                self.assertIn("signals//loss/reg", losses)
                self.assertIn("signals//loss/total", losses)
                self.assertIn("signals//accuracy", losses)
                self.assertIn("signals//f1_score", losses)

    def test_signal_naming_with_nested_paths(self):
        """Test that signal names with nested paths (slashes) are handled correctly."""
        @wl.signal(name="train/loss/weighted")
        def nested_signal(x):
            return x

        @wl.signal(name="eval/metrics/precision")
        def nested_signal2(x):
            return x

        self.assertIn("train/loss/weighted", _REGISTERED_SIGNALS)
        self.assertIn("eval/metrics/precision", _REGISTERED_SIGNALS)

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_signals_with_different_batch_sizes(self, mock_gm, mock_get_dataframe):
        """Test save_signals with different batch sizes."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            # Small batch
            wl.save_signals(
                signals={"loss": torch.tensor(0.5)},
                batch_ids=torch.tensor([1]),
                log=True
            )

            # Large batch
            wl.save_signals(
                signals={"loss": torch.tensor(0.4)},
                batch_ids=torch.tensor([i for i in range(256)]),
                log=True
            )

            self.assertEqual(mock_df.enqueue_batch.call_count, 2)

    # =========================================================================
    # Detection-Specific Signal Tests
    # =========================================================================

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_detection_signals_with_variable_boxes(self, mock_gm, mock_get_dataframe):
        """Test detection signals with variable number of bounding boxes per image."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2, 3, 4])
            signals = {
                "loss/bbox": torch.tensor(0.25),
                "loss/cls": torch.tensor(0.15),
                "loss/dfl": torch.tensor(0.10)
            }

            # Variable number of boxes: img1 has 3, img2 has 1, img3 has 5, img4 has 2
            preds = [
                torch.tensor([[10, 20, 110, 120], [50, 60, 150, 160], [200, 210, 300, 310]]),  # 3 boxes
                torch.tensor([[15, 25, 115, 125]]),  # 1 box
                torch.tensor([[30, 40, 130, 140], [70, 80, 170, 180], [250, 260, 350, 360],
                             [100, 110, 200, 210], [180, 190, 280, 290]]),  # 5 boxes
                torch.tensor([[45, 55, 145, 155], [220, 230, 320, 330]])  # 2 boxes
            ]

            targets = [
                torch.tensor([[12, 22, 112, 122], [52, 62, 152, 162], [202, 212, 302, 312]]),
                torch.tensor([[17, 27, 117, 127]]),
                torch.tensor([[32, 42, 132, 142], [72, 82, 172, 182], [252, 262, 352, 362],
                             [102, 112, 202, 212], [182, 192, 282, 292]]),
                torch.tensor([[47, 57, 147, 157], [222, 232, 322, 332]])
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
            self.assertEqual(len(call_kwargs['sample_ids']), 4)

    # =========================================================================
    # Integration Tests with Wrapped Models
    # =========================================================================

    def test_signal_metadata_inheritance(self):
        """Test that signal metadata is properly inherited and accessible."""
        @wl.signal(
            name="comprehensive_signal",
            subscribe_to="training_loss",
            compute_every_n_steps=5,
            custom_param="custom_value",
            priority=10
        )
        def signal_func(x):
            return x

        meta = signal_func._wl_signal_meta
        self.assertEqual(meta['subscribe_to'], "training_loss")
        self.assertEqual(meta['compute_every_n_steps'], 5)
        self.assertEqual(meta['custom_param'], "custom_value")
        self.assertEqual(meta['priority'], 10)
        self.assertEqual(signal_func._wl_signal_name, "comprehensive_signal")

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_signals_for_multiclass_classification(self, mock_gm, mock_get_dataframe):
        """Test signals for multi-class classification task."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([1, 2, 3, 4, 5])
            signals = {
                "loss": torch.tensor(0.45),
                "top1_accuracy": torch.tensor(0.88),
                "top5_accuracy": torch.tensor(0.98)
            }
            # Multi-class logits (5 samples, 10 classes)
            preds = torch.rand(5, 10)
            targets = torch.randint(0, 10, (5,))

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            self.assertIn("signals//loss", call_kwargs['losses'])
            self.assertIn("signals//top1_accuracy", call_kwargs['losses'])

    @patch("weightslab.src.get_dataframe")
    @patch("weightslab.src._gm")
    def test_signals_for_binary_classification(self, mock_gm, mock_get_dataframe):
        """Test signals for binary classification task."""
        mock_df = MagicMock()
        mock_get_dataframe.return_value = mock_df

        mock_model = MagicMock()
        mock_model.current_step = 1
        mock_gm.return_value = mock_model

        with patch("weightslab.src.DATAFRAME_M", mock_df):
            batch_ids = torch.tensor([10, 11, 12, 13, 14])
            signals = {
                "loss": torch.tensor(0.35),
                "accuracy": torch.tensor(0.92),
                "precision": torch.tensor(0.91),
                "recall": torch.tensor(0.90),
                "auc": torch.tensor(0.95)
            }
            # Binary classification probabilities
            preds = torch.sigmoid(torch.randn(5))
            targets = torch.randint(0, 2, (5,), dtype=torch.float32)

            wl.save_signals(
                signals=signals,
                batch_ids=batch_ids,
                preds=preds,
                targets=targets,
                log=True
            )

            self.assertTrue(mock_df.enqueue_batch.called)
            call_kwargs = mock_df.enqueue_batch.call_args[1]
            losses = call_kwargs['losses']
            self.assertEqual(len(losses), 5)  # All signals should be saved

if __name__ == "__main__":
    unittest.main()
