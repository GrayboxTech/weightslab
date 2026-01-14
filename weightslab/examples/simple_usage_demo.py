"""
Simple Usage Demo: H5 Array Storage - No Manual Conversion Needed!

This demonstrates that with LedgeredDataFrameManager, you NEVER need to
call convert_dataframe_to_proxies() or any other conversion function.
Everything happens automatically!
"""

import numpy as np
from pathlib import Path
import tempfile

from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.data.dataframe_manager import LedgeredDataFrameManager


def main():
    print("="*80)
    print("Simple Usage Demo: Zero Manual Conversion Required!")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ============================================================
        # STEP 1: Setup (one-time initialization)
        # ============================================================
        print("\n1. Initialize (one-time setup):")

        ledger = LedgeredDataFrameManager(
            flush_interval=1.0,
            flush_max_rows=10,
            enable_h5_persistence=True
        )
        main_store = H5DataFrameStore(tmpdir / "experiment.h5")
        ledger.set_store(main_store)

        print("   ✓ LedgeredDataFrameManager created")
        print(f"   ✓ Main H5: {main_store.get_path()}")
        print(f"   ✓ Arrays H5: {ledger.get_array_store().get_path()}")

        # ============================================================
        # STEP 2: Training Loop (store arrays automatically)
        # ============================================================
        print("\n2. During training - just enqueue batches:")

        sample_ids = [1, 2, 3, 4, 5]
        predictions = (np.random.rand(5, 128, 128, 3) > 0.5).astype(np.uint8)
        predictions_raw = np.random.rand(5, 128, 128, 3).astype(np.float32)
        targets = np.random.randint(0, 10, size=(5, 128, 128)).astype(np.int32)

        ledger.enqueue_batch(
            model_age=1,
            sample_ids=sample_ids,
            preds=predictions,
            targets=targets,
            preds_raw=predictions_raw,
            losses={"loss": np.array([0.5, 0.4, 0.3, 0.2, 0.1])}
        )
        ledger.flush_if_needed(force=True)

        print(f"   ✓ Stored {len(sample_ids)} samples")
        print("   ✓ Arrays automatically saved to arrays.h5")
        print("   ✓ Paths automatically saved to main H5")

        # ============================================================
        # STEP 3: Analysis - just get dataframe and use it!
        # ============================================================
        print("\n3. Analysis - NO manual conversion needed:")

        # Eager-load all array columns so df.loc returns numpy arrays
        df = ledger.get_combined_df(
            autoload_arrays=True,
            return_proxies=False
        )

        print("\n   Just use df.loc like normal pandas:")
        print("   " + "-" * 50)

        # Access arrays - they're automatically loaded!
        for sample_id in [1, 2, 3]:
            pred = df.loc[sample_id, 'prediction']
            predictions_raw = df.loc[sample_id, 'predictions_raw']
            target = df.loc[sample_id, 'target']
            loss = df.loc[sample_id, 'loss']

            print(f"\n   Sample {sample_id}:")
            print(f"     • prediction: {type(pred).__name__}{pred.shape}")
            print(f"     • target:     {type(target).__name__}{target.shape}")
            print(f"     • loss:       {loss:.4f}")

        # ============================================================
        # STEP 4: Do normal numpy operations
        # ============================================================
        print("\n4. Work with arrays normally:")

        pred = df.loc[1, 'prediction']   # numpy array
        target = df.loc[1, 'target']     # numpy array

        # Work with arrays directly
        mean_pred = pred.mean()
        std_pred = pred.std()
        unique_targets = np.unique(target)

        print(f"   • Mean prediction: {mean_pred:.4f}")
        print(f"   • Std prediction:  {std_pred:.4f}")
        print(f"   • Unique targets:  {unique_targets[:5]}...")

        # ============================================================
        # OPTIONAL: Lazy loading for very large datasets
        # ============================================================
        print("\n5. (Optional) Lazy loading for huge datasets:")

        df_lazy = ledger.get_combined_df(
            autoload_arrays=False,  # keep proxies
            return_proxies=True
        )

        proxy = df_lazy.loc[1, 'prediction']
        print(f"   • Lightweight proxy: {proxy}")
        print(f"   • Load when needed: {proxy.load().shape}")
        print(f"   • Memory saved: Arrays stay on disk until accessed")

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "="*80)
        print("SUMMARY: What you DON'T need to do:")
        print("="*80)
        print("  ✗ No convert_dataframe_to_proxies() calls")
        print("  ✗ No manual array store management")
        print("  ✗ No special accessor methods (df.arrays.load_sample)")
        print("  ✗ No path-to-array conversion code")
        print()
        print("What you DO:")
        print("="*80)
        print("  ✓ df = ledger.get_combined_df()")
        print("  ✓ array = df.loc[sample_id, 'prediction']")
        print("  ✓ That's it!")
        print("="*80)


if __name__ == "__main__":
    main()
