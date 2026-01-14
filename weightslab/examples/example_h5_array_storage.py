"""
Example: Using the new H5 Array Storage System in Weightslab

This example demonstrates:
1. How arrays are automatically stored in a separate arrays.h5 file
2. How path references are stored in the main dataframe
3. How to lazily load arrays using ArrayH5Proxy
4. How to use the pandas accessor for convenient array access
5. How to configure uint8 normalization for space savings
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.h5_array_store import H5ArrayStore
from weightslab.data.array_proxy import ArrayH5Proxy, convert_dataframe_to_proxies
from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.sample_stats import SampleStats


def example_1_basic_array_storage():
    """Example 1: Basic array storage and retrieval"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Array Storage")
    print("="*80)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize array store
        array_store = H5ArrayStore(tmpdir / "arrays.h5", auto_normalize=True)

        # Create some sample arrays
        sample_id = 123
        prediction = np.random.rand(224, 224, 3).astype(np.float32)  # Image-like prediction
        target = np.random.randint(0, 10, size=(224, 224)).astype(np.int32)  # Segmentation mask

        print(f"\n1. Saving arrays for sample_id={sample_id}")
        print(f"   - Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")
        print(f"   - Target shape: {target.shape}, dtype: {target.dtype}")

        # Save arrays and get path references
        pred_path = array_store.save_array(sample_id, "prediction", prediction)
        target_path = array_store.save_array(sample_id, "target", target)

        print(f"\n2. Path references generated:")
        print(f"   - Prediction: {pred_path}")
        print(f"   - Target: {target_path}")

        # Load arrays back
        print(f"\n3. Loading arrays back from storage...")
        loaded_pred = array_store.load_array(pred_path)
        loaded_target = array_store.load_array(target_path)

        print(f"   - Loaded prediction shape: {loaded_pred.shape}, dtype: {loaded_pred.dtype}")
        print(f"   - Loaded target shape: {loaded_target.shape}, dtype: {loaded_target.dtype}")

        # Verify data integrity (with tolerance for uint8 normalization)
        pred_diff = np.abs(prediction - loaded_pred).max()
        print(f"\n4. Data integrity check:")
        print(f"   - Max prediction difference: {pred_diff:.6f}")
        print(f"   - Note: Some loss expected due to uint8 normalization for space savings")

        # Show file sizes
        array_file_size = (tmpdir / "arrays.h5").stat().st_size / 1024  # KB
        print(f"\n5. Storage efficiency:")
        print(f"   - arrays.h5 file size: {array_file_size:.2f} KB")


def example_2_batch_operations():
    """Example 2: Batch array storage and retrieval"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Operations")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        array_store = H5ArrayStore(tmpdir / "arrays.h5")

        # Create batch of arrays
        num_samples = 5
        arrays_dict = {}

        print(f"\n1. Creating batch of {num_samples} samples...")
        for i in range(num_samples):
            sample_id = 100 + i
            arrays_dict[sample_id] = {
                "prediction": np.random.rand(64, 64, 3).astype(np.float32),
                "target": np.random.randint(0, 10, size=(64, 64)).astype(np.int32),
                "prediction_raw": np.random.rand(64, 64, 10).astype(np.float32)  # Logits
            }

        # Batch save
        print(f"\n2. Batch saving arrays...")
        path_refs = array_store.save_arrays_batch(arrays_dict, preserve_original=False)

        print(f"\n3. Generated path references for {len(path_refs)} samples:")
        for sample_id, refs in list(path_refs.items())[:2]:  # Show first 2
            print(f"   Sample {sample_id}:")
            for key, path in refs.items():
                print(f"      - {key}: {path}")

        # Batch load
        print(f"\n4. Batch loading arrays...")
        loaded_arrays = array_store.load_arrays_batch(path_refs)

        print(f"\n5. Loaded {len(loaded_arrays)} samples successfully")
        sample_id = list(loaded_arrays.keys())[0]
        print(f"   Sample {sample_id} contains keys: {list(loaded_arrays[sample_id].keys())}")


def example_3_dataframe_integration():
    """Example 3: Typical workflow with LedgeredDataFrameManager"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Typical Workflow - Just Use get_combined_df()")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Setup ledger manager (typical use case)
        ledger = LedgeredDataFrameManager(
            flush_interval=1.0,
            flush_max_rows=5,
            enable_h5_persistence=True
        )
        main_store = H5DataFrameStore(tmpdir / "experiment.h5")
        ledger.set_store(main_store)

        print("\n1. Store arrays during training (typical workflow)...")
        sample_ids = [200, 201, 202]
        predictions = np.random.rand(3, 32, 32, 3).astype(np.float32)
        targets = np.random.randint(0, 5, size=(3, 32, 32)).astype(np.int32)

        ledger.enqueue_batch(
            model_age=1,
            sample_ids=sample_ids,
            preds=predictions,
            targets=targets,
            preds_raw=None,
            losses={"loss": np.array([0.5, 0.4, 0.3])}
        )
        ledger.flush_if_needed(force=True)
        print("   ✓ Arrays automatically saved to arrays.h5")

        print("\n2. Get dataframe and access arrays - just use df.loc:")
        df = ledger.get_combined_df()  # That's it! Arrays auto-loaded

        sample_id = sample_ids[0]
        prediction = df.loc[sample_id, "prediction"]
        target = df.loc[sample_id, "target"]
        loss = df.loc[sample_id, "loss"]

        print(f"   - prediction type: {type(prediction)}")
        print(f"   - prediction shape: {prediction.shape}")
        print(f"   - target shape: {target.shape}")
        print(f"   - loss (scalar): {loss}")

        print("\n3. (Optional) Lazy loading for very large datasets:")
        df_lazy = ledger.get_combined_df(
            autoload_arrays=False,  # Keep as lightweight proxies
            return_proxies=True
        )

        proxy = df_lazy.loc[sample_id, "prediction"]
        print(f"   - Type: {type(proxy)}")
        print(f"   - Repr: {proxy}")
        print(f"   - Load on demand: {proxy.load().shape}")


def example_4_ledger_manager_integration():
    """Example 4: Full integration with LedgeredDataFrameManager"""
    print("\n" + "="*80)
    print("EXAMPLE 4: LedgeredDataFrameManager Integration")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create stores
        main_store = H5DataFrameStore(tmpdir / "data_with_ops.h5")

        # Create ledger manager
        print("\n1. Initializing LedgeredDataFrameManager...")
        ledger = LedgeredDataFrameManager(
            flush_interval=1.0,
            flush_max_rows=10,
            enable_h5_persistence=True
        )
        ledger.set_store(main_store)  # This auto-creates array_store

        print(f"   - Main H5: {main_store.path}")
        print(f"   - Array H5: {ledger.get_array_store().path}")

        # Simulate training loop
        print("\n2. Simulating training batch updates...")
        sample_ids = [1, 2, 3]
        predictions = np.random.rand(3, 64, 64, 3).astype(np.float32)
        targets = np.random.randint(0, 5, size=(3, 64, 64)).astype(np.int32)
        preds_raw = np.random.rand(3, 64, 64, 5).astype(np.float32)

        losses = {
            "total_loss": np.array([0.5, 0.4, 0.3]),
            "ce_loss": np.array([0.3, 0.25, 0.2])
        }

        # Enqueue batch
        ledger.enqueue_batch(
            model_age=1,
            sample_ids=sample_ids,
            preds_raw=preds_raw,
            preds=predictions,
            losses=losses,
            targets=targets
        )

        print(f"   - Enqueued {len(sample_ids)} samples")

        # Force flush
        print("\n3. Flushing to disk...")
        ledger.flush_if_needed(force=True)

        # Get combined dataframe - arrays automatically converted!
        print("\n4. Retrieving combined dataframe (auto-converts arrays)...")
        df = ledger.get_combined_df()  # That's it! No extra parameters needed

        print(f"\n   DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        print(f"\n5. Access arrays directly with df.loc - no conversion needed!:")
        sample_id = sample_ids[0]
        if sample_id in df.index:
            # By default, get_combined_df() auto-loads arrays
            prediction = df.loc[sample_id, SampleStats.Ex.PREDICTION.value]
            target = df.loc[sample_id, SampleStats.Ex.TARGET.value]
            loss = df.loc[sample_id, 'total_loss']

            print(f"   - prediction type: {type(prediction)}")
            print(f"   - prediction shape: {prediction.shape}")
            print(f"   - target shape: {target.shape}")
            print(f"   - loss (scalar): {loss}")

        print(f"\n6. Lazy loading option (defer loading until needed):")
        df_lazy = ledger.get_combined_df(
            autoload_arrays=False,  # Keep as proxies/paths
            return_proxies=True     # Convert to proxies
        )

        if sample_id in df_lazy.index:
            pred_proxy = df_lazy.loc[sample_id, SampleStats.Ex.PREDICTION.value]
            print(f"   - Type: {type(pred_proxy)}")
            print(f"   - Repr: {pred_proxy}")
            print(f"   - Array loaded on demand: {pred_proxy.load().shape}")


def example_5_uint8_normalization():
    """Example 5: Demonstrating uint8 normalization for space savings"""
    print("\n" + "="*80)
    print("EXAMPLE 5: uint8 Normalization")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test array
        original = np.random.rand(256, 256, 3).astype(np.float32)
        original_size = original.nbytes / 1024  # KB

        print(f"\n1. Original array:")
        print(f"   - Shape: {original.shape}")
        print(f"   - Dtype: {original.dtype}")
        print(f"   - Memory size: {original_size:.2f} KB")
        print(f"   - Value range: [{original.min():.4f}, {original.max():.4f}]")

        # Store with auto-normalization (default)
        print(f"\n2. Storing with auto uint8 normalization...")
        array_store_normalized = H5ArrayStore(tmpdir / "arrays_normalized.h5", auto_normalize=True)
        path_ref = array_store_normalized.save_array(1, "test", original, preserve_original=False)

        # Store without normalization
        print(f"\n3. Storing without normalization (preserve original)...")
        array_store_original = H5ArrayStore(tmpdir / "arrays_original.h5", auto_normalize=False)
        path_ref2 = array_store_original.save_array(1, "test", original, preserve_original=True)

        # Compare file sizes
        size_normalized = (tmpdir / "arrays_normalized.h5").stat().st_size / 1024
        size_original = (tmpdir / "arrays_original.h5").stat().st_size / 1024

        print(f"\n4. Storage comparison:")
        print(f"   - Normalized file size: {size_normalized:.2f} KB")
        print(f"   - Original file size: {size_original:.2f} KB")
        print(f"   - Space savings: {(1 - size_normalized/size_original)*100:.1f}%")

        # Load and check accuracy
        loaded_normalized = array_store_normalized.load_array(path_ref)
        loaded_original = array_store_original.load_array(path_ref2)

        print(f"\n5. Data integrity:")
        print(f"   - Normalized max error: {np.abs(original - loaded_normalized).max():.6f}")
        print(f"   - Original max error: {np.abs(original - loaded_original).max():.6f}")
        print(f"   - Normalized RMSE: {np.sqrt(np.mean((original - loaded_normalized)**2)):.6f}")


def example_6_typical_usage():
    """Example 6: Typical usage patterns"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Typical Usage - No Extra Code Needed!")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Setup
        ledger = LedgeredDataFrameManager(
            flush_interval=1.0,
            flush_max_rows=10,
            enable_h5_persistence=True
        )
        main_store = H5DataFrameStore(tmpdir / "data.h5")
        ledger.set_store(main_store)

        print("\n1. Store data during training (automatic):")
        sample_ids = [1, 2, 3]
        predictions = np.random.rand(3, 64, 64, 3).astype(np.float32)
        targets = np.random.randint(0, 5, size=(3, 64, 64)).astype(np.int32)

        ledger.enqueue_batch(
            model_age=1,
            sample_ids=sample_ids,
            preds=predictions,
            targets=targets,
            preds_raw=None,
            losses={"loss": np.array([0.5, 0.4, 0.3])}
        )
        ledger.flush_if_needed(force=True)
        print("   ✓ Arrays automatically stored in arrays.h5")

        print("\n2. Access data - just call get_combined_df() and use df.loc:")
        df = ledger.get_combined_df()  # Arrays automatically loaded!

        # Just use df.loc - that's it!
        pred_array = df.loc[1, 'prediction']
        target_array = df.loc[1, 'target']
        loss_value = df.loc[1, 'loss']

        print(f"   - df.loc[1, 'prediction'] → {type(pred_array).__name__}{pred_array.shape}")
        print(f"   - df.loc[1, 'target'] → {type(target_array).__name__}{target_array.shape}")
        print(f"   - df.loc[1, 'loss'] → {loss_value}")

        print("\n3. Work with arrays normally:")
        # No special handling needed!
        mean_pred = pred_array.mean()
        target_unique = np.unique(target_array)

        print(f"   - Mean prediction: {mean_pred:.4f}")
        print(f"   - Unique targets: {target_unique}")

        print("\n4. For very large datasets, use lazy loading:")
        df_lazy = ledger.get_combined_df(
            autoload_arrays=False,
            return_proxies=True
        )

        proxy = df_lazy.loc[1, 'prediction']
        print(f"   - Lightweight proxy: {proxy}")
        print(f"   - Load when needed: proxy.load() → {proxy.load().shape}")

        print("\n✓ That's it! No accessors, no special methods - just df.loc!")


def main():
    """Run all examples"""
    print("\n" + "#"*80)
    print("# Weightslab H5 Array Storage System - Examples")
    print("#"*80)

    example_1_basic_array_storage()
    example_2_batch_operations()
    example_3_dataframe_integration()
    example_4_ledger_manager_integration()
    example_5_uint8_normalization()
    example_6_typical_usage()

    print("\n" + "#"*80)
    print("# All examples completed successfully!")
    print("#"*80)
    print("\nKey Takeaways:")
    print("1. Arrays automatically stored in arrays.h5, paths in main H5")
    print("2. get_combined_df() automatically converts paths → arrays (no manual calls!)")
    print("3. Just use df.loc[sample_id, 'prediction'] - works like normal pandas!")
    print("4. uint8 normalization saves ~75% storage with minimal accuracy loss")
    print("5. For large datasets: use autoload_arrays=False to defer loading")
    print("6. Zero extra code needed - it just works!")
    print()


if __name__ == "__main__":
    main()
