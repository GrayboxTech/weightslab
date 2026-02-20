"""
Backend: Data loader interface for wrapping datasets and dataloaders.

This module provides a lightweight interface around PyTorch datasets and
`torch.utils.data.DataLoader` instances used by WeightsLab. It is intentionally
small: it normalizes inputs (dataset or DataLoader), exposes convenience
methods used by the rest of the codebase (like `as_records`) and provides a
resettable iterator and a safe `next_batch()` helper.

It also supports:
- a mutable batch sampler for runtime batch-size changes
- masked sampling that automatically excludes deny-listed samples during iteration
- global pause control
- registration in a global ledger and dynamic batch-size updates based on
  hyperparameters
- checkpoint-based data loading and reproducible iterator restoration
"""
import torch
import logging
from typing import Any, Iterator, Optional

from torch.utils.data import DataLoader, Dataset, Sampler

from weightslab.data.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.utils import filter_kwargs_for_callable, restore_rng_state
from weightslab.components.global_monitoring import pause_controller
from weightslab.backend.ledgers import (
    register_dataloader,
    get_hyperparams,
    resolve_hp_name,
    get_checkpoint_manager,
)
from weightslab.data.sample_stats import SampleStatsEx


# Get Global Logger
logger = logging.getLogger(__name__)


class WeightsLabDataSampler(Sampler):
    """Unified sampler for WeightsLab with shuffle, masking, offset, and optional batching.

    This sampler combines the functionality of multiple samplers:
    - Shuffle/sequential mode toggle (runtime changeable)
    - Deny-list filtering for tracked datasets
    - Sample offset for checkpoint restoration
    - Optional batching with mutable batch size

    Can be used as either a regular sampler (yields indices) or batch sampler
    (yields lists of indices) based on batch_size parameter.

    Example:
        # As a regular sampler with shuffle and masking
        sampler = WeightsLabDataSampler(dataset, tracked_dataset=wrapper, shuffle=True)
        loader = DataLoader(dataset, sampler=sampler)

        # As a batch sampler with all features
        sampler = WeightsLabDataSampler(
            dataset, tracked_dataset=wrapper, shuffle=True,
            offset=100, batch_size=32, drop_last=True
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

        # Toggle shuffle at runtime
        sampler.shuffle = False  # Switch to sequential
    """

    def __init__(
        self,
        data_source: Dataset,
        tracked_dataset: Optional[DataSampleTrackingWrapper] = None,
        shuffle: bool = True,
        offset: int = 0,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
    ):
        """Initialize the unified sampler.

        Args:
            data_source: The dataset to sample from
            tracked_dataset: Optional tracking wrapper for deny-list support
            shuffle: Whether to shuffle indices during iteration
            offset: Number of samples to skip from the beginning
            batch_size: If provided, acts as batch sampler yielding lists of indices
            drop_last: If True, drop incomplete batches (only used if batch_size is set)
        """
        self.data_source = data_source
        self.tracked_dataset = tracked_dataset or data_source
        self.shuffle = shuffle
        self.offset = max(0, offset)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _get_deny_listed_uids(self) -> set:
        """Get set of deny-listed UIDs from tracked dataset."""
        deny_listed_uids = set()
        if self.tracked_dataset is not None and hasattr(self.tracked_dataset, "_get_df_view"):
            try:
                df_view = self.tracked_dataset._get_df_view()
                if not df_view.empty and SampleStatsEx.DISCARDED.value in df_view.columns:
                    deny_listed_uids = set(df_view[df_view[SampleStatsEx.DISCARDED.value] == True].index)
            except Exception:
                pass
        return deny_listed_uids

    def _generate_indices(self):
        """Generate base indices (shuffled or sequential)."""
        n = len(self.data_source)
        if self.shuffle:
            indices = torch.randperm(n).tolist()
        else:
            indices = list(range(n))
        return indices

    def _filter_indices(self, indices):
        """Filter out deny-listed and offset indices."""
        deny_listed_uids = self._get_deny_listed_uids()

        # Filter and apply offset
        filtered = []
        skipped = 0

        for idx in indices:
            # Check if deny-listed
            if self.tracked_dataset is not None and hasattr(self.tracked_dataset, "unique_ids"):
                try:
                    uid = int(self.tracked_dataset.unique_ids[idx])
                    if uid in deny_listed_uids:
                        continue
                except Exception:
                    pass

            # Apply offset
            if skipped < self.offset:
                skipped += 1
                continue

            filtered.append(idx)

        return filtered

    def __iter__(self):
        """Iterate over indices or batches of indices."""
        # Generate and filter indices
        indices = self._generate_indices()
        filtered_indices = self._filter_indices(indices)

        # If no batching, yield individual indices
        if self.batch_size is None:
            yield from filtered_indices
        else:
            # Yield batches
            batch = []
            for idx in filtered_indices:
                batch.append(idx)
                if len(batch) >= int(self.batch_size):
                    yield list(batch)
                    batch = []

            # Yield remaining batch if not dropping last
            if batch and not self.drop_last:
                yield list(batch)

    def __len__(self):
        """Return the number of samples or batches."""
        # Start with total dataset size
        total = len(self.data_source)

        # Subtract deny-listed samples
        deny_listed_uids = self._get_deny_listed_uids()
        total -= len(deny_listed_uids)

        # Subtract offset
        total = max(0, total - self.offset)

        # If batching, return number of batches
        if self.batch_size is not None:
            b = max(1, int(self.batch_size))
            if self.drop_last:
                return total // b
            else:
                return (total + b - 1) // b

        return total


class DataLoaderInterface:
    """Wrap a Dataset or DataLoader and expose common helpers.

    The interface accepts either a `Dataset` instance or a pre-built
    `DataLoader`. When given a `Dataset` it will build a `DataLoader` using
    provided kwargs. The wrapped dataset is exposed as `self.dataset` and the
    dataloader as `self.dataloader`.

    The rest of WeightsLab expects datasets to implement an `as_records()`
    method (see `weightslab.data.data_samples_with_ops.DataSampleTrackingWrapper`).
    If the wrapped dataset provides `as_records()` we delegate to it; otherwise
    `as_records()` will raise `AttributeError`.

    **Deny-listed Sample Handling:**
    When using a DataSampleTrackingWrapper, deny-listed samples are automatically
    excluded from batches during iteration via the WeightsLabDataSampler. This allows
    you to dynamically filter samples at train/eval time without rebuilding
    the dataset:

        dataset.denylist_samples({sample_id_1, sample_id_2, ...})
        # Next iteration will skip these samples automatically

    Public API:
    - __iter__, __len__, __next__
    - next_batch(), reset_iterator()
    - as_records()
    - set_transform()
    - get_dataloader()
    - batch_size / get_batch_size(), set_batch_size()
    """

    def __init__(
        self,
        data_loader_or_dataset: Any,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = True,
        collate_fn: Optional[Any] = None,
        loader_name: Optional[str] = None,
        register: bool = True,
        weak: bool = False,
        root_log_dir: Optional[str] = None,
        compute_hash: bool = True,
        use_tags: bool = False,
        tags_mapping: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initialize the DataLoaderInterface.
        Args:
            data_loader_or_dataset: A Dataset or DataLoader instance to wrap.
            batch_size: Batch size for DataLoader if a Dataset is provided.
            shuffle: Whether to shuffle data if a Dataset is provided.
            num_workers: Number of worker processes for DataLoader.
            drop_last: Whether to drop the last incomplete batch.
            pin_memory: Whether to use pinned memory for DataLoader.
            collate_fn: Optional collate function for DataLoader.
            loader_name: Optional name for registration in the global ledger.
            register: Whether to register this interface in the global ledger.
            weak: Whether to use weak references when registering.
            root_log_dir: Optional root log directory for tracking wrapper.
            compute_hash: Whether to compute hashes for samples in tracking wrapper.
            use_tags: Whether to use tags for samples in tracking wrapper.
            tags_mapping: Optional mapping of tags to integer labels.

            **kwargs: Additional kwargs passed to DataLoader if a Dataset is provided.
        """
        # Normalize inputs
        self.dataset: Dataset | DataLoader = data_loader_or_dataset

        # Internal flags / helpers
        self._mutable_batch_sampler = None
        self._dl_build_kwargs: Optional[dict] = None
        self._pending_iteration_state: Optional[dict] = None
        self.is_training = kwargs.pop("is_training", False)
        self._enable_h5_persistence = kwargs.pop("enable_h5_persistence", True)

        if isinstance(data_loader_or_dataset, DataLoader):
            logger.warning(
                "DataLoaderInterface: wrapping user-supplied DataLoader !! Highly experimental, user should ensure compatibility !! "
                "Otherwise, prefer passing a Dataset and let the interface build the DataLoader."
                )
            # User-supplied dataloader
            self.dataloader: DataLoader = data_loader_or_dataset
            self.tracked_dataset = DataSampleTrackingWrapper(
                self.dataloader.dataset if hasattr(self.dataloader, "dataset") else self.dataloader,
                root_log_dir=root_log_dir,
                compute_hash=compute_hash,
                use_tags=use_tags,
                tags_mapping=tags_mapping,
                loader_name=loader_name,
                **kwargs
            )
            self.tracked_dataset._map_updates_hook_fns.append(
                (
                    self._reset_iterator,
                    {}
                )
            )
        else:
            # First, wrap the dataset with our tracking wrapper to get deny-list and logging support
            self.tracked_dataset = DataSampleTrackingWrapper(
                data_loader_or_dataset,
                root_log_dir=root_log_dir,
                compute_hash=compute_hash,
                use_tags=use_tags,
                tags_mapping=tags_mapping,
                loader_name=loader_name,
                **kwargs
            )
            self.tracked_dataset._map_updates_hook_fns.append(
                (
                    self._reset_iterator,
                    {}
                )
            )

            # Then, load checkpoint data early (before dataloader is used)
            self._load_checkpoint_data()

            # Next, define the batch sampler with the initial offset (if any) for checkpoint restoration
            batch_sampler = WeightsLabDataSampler(
                self.tracked_dataset,
                tracked_dataset=self.tracked_dataset,
                shuffle=shuffle,
                offset=0,
                batch_size=batch_size,
                drop_last=drop_last,
            )

            # Finally, construct dataloader using our batch_sampler
            self.dataloader = DataLoader(
                self.tracked_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                **filter_kwargs_for_callable(DataLoader, kwargs)
            )
            self._mutable_batch_sampler = batch_sampler

            # Store kwargs so we can recreate dataloader if needed
            self._dl_build_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "drop_last": drop_last,
                "pin_memory": pin_memory,
                "collate_fn": collate_fn,
            }
            self._dl_build_kwargs.update(kwargs or {})

        self._init_attributes(self.dataloader)

        # Apply pending iteration state if one was loaded from checkpoint
        if hasattr(self, '_pending_iteration_state') and self._pending_iteration_state:
            try:
                self.restore_iteration_state(self._pending_iteration_state)
                logger.info(f"Restored dataloader iteration state: {self._pending_iteration_state}")
            except Exception as e:
                logger.warning(f"Failed to restore pending iteration state: {e}")
            finally:
                self._pending_iteration_state = None

        # Internal iterator used by `_next_batch` (lazy created to avoid consuming RNG early)
        self._iterator: Optional[Iterator] = None
        # Track how many samples have been yielded since last reset (for reproducible seeking)
        self._samples_yielded: int = 0
        self._sample_offset: int = 0
        self._skipped = []

        # Optionally register in the global ledger for cross-thread access.
        # If no explicit `loader_name` is provided, try to infer a friendly loader_name from
        # the wrapped dataset class loader_name; otherwise fall back to '_dataloader'.
        self._ledger_name = None
        if register:
            self._ledger_name = loader_name
            try:
                register_dataloader(self, weak=weak, name=loader_name)
            except Exception:
                # Best-effort: ignore registration failures
                pass

    def _load_checkpoint_data(self) -> None:
        """Load data checkpoint, RNG state, and dataloader iteration state early.

        This method is called after tracked_dataset initialization to restore
        data from the latest checkpoint if available. It:
        1. Loads data snapshot and applies it to the dataframe
        2. Restores RNG state for reproducible shuffling
        3. Restores dataloader iteration state for deterministic seeking
        """
        try:
            checkpoint_manager = get_checkpoint_manager()
            if checkpoint_manager is None:
                return

            # Get latest experiment hash
            latest_hash = None
            if hasattr(checkpoint_manager, 'current_exp_hash') and checkpoint_manager.current_exp_hash:
                latest_hash = checkpoint_manager.current_exp_hash
            elif hasattr(checkpoint_manager, 'get_latest_hash'):
                latest_hash = checkpoint_manager.get_latest_hash()

            if not latest_hash:
                return

            # Load checkpoint with data state
            checkpoint_data = checkpoint_manager.load_checkpoint(
                exp_hash=latest_hash,
                load_model=False,
                load_weights=False,
                load_config=False,
                load_data=True,
                force=True
            )

            if not checkpoint_data.get('loaded_components'):
                return

            # Apply data snapshot to dataframe if loaded
            if 'data' in checkpoint_data['loaded_components']:
                try:
                    data_state = checkpoint_data.get('data_state', {})
                    snapshot_df = data_state.get('snapshot')

                    if snapshot_df is not None and not snapshot_df.empty:
                        if hasattr(self.tracked_dataset, 'upsert_df'):
                            self.tracked_dataset.upsert_df(snapshot_df, force_flush=True)
                            logger.info(f"Applied data snapshot from checkpoint ({len(snapshot_df)} rows)")
                except Exception as e:
                    logger.warning(f"Failed to apply data snapshot: {e}")

            # Restore RNG state for reproducible shuffling
            if checkpoint_data.get('rng_state'):
                try:
                    restore_rng_state(checkpoint_data['rng_state'])
                    logger.debug("Restored RNG state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to restore RNG state: {e}")

            # Restore dataloader iteration state for deterministic seeking
            if checkpoint_data.get('dataloader_iteration_state'):
                try:
                    iter_state = checkpoint_data['dataloader_iteration_state']
                    # Normalize to handle both dict and single state formats
                    if isinstance(iter_state, dict) and 'samples_yielded' in iter_state:
                        # Single state format; will be applied when dataloader is ready
                        self._pending_iteration_state = iter_state
                    elif isinstance(iter_state, dict):
                        # Multi-loader format; pick one for this loader
                        state_for_loader = iter_state.get(self._ledger_name) or iter_state.get('default') or next(iter(iter_state.values()), None)
                        if state_for_loader:
                            self._pending_iteration_state = state_for_loader
                    else:
                        self._pending_iteration_state = iter_state

                    if hasattr(self, '_pending_iteration_state'):
                        logger.debug(f"Pending iteration state to restore: {self._pending_iteration_state}")
                except Exception as e:
                    logger.warning(f"Failed to parse dataloader iteration state: {e}")

        except Exception as e:
            logger.debug(f"Could not load checkpoint data: {e}")

    def _init_attributes(self, obj):
        """Expose attributes and methods from the wrapped `obj`.

        Implementation strategy (direct iteration):
        - Iterate over `vars(obj)` to obtain instance attributes and
          create class-level properties that forward to `obj.<attr>`.
        - Iterate over `vars(obj.__class__)` to find callables (methods)
          and bind the object's bound method to this wrapper instance so
          calling `iface.method()` invokes `iface.obj.method()`.

        This avoids using `dir()` and directly inspects the object's
        own dictionaries. Existing attributes on DataLoaderInterface are
        preserved and not overwritten.
        """
        if obj is None:
            return

        # Existing names on the wrapper instance/class to avoid overwriting
        existing_instance_names = set(self.__dict__.keys())
        existing_class_names = set(getattr(self.__class__, '__dict__', {}).keys())

        # 1) Expose instance attributes of `obj` as properties on the wrapper class
        obj_vars = getattr(obj, '__dict__', {})
        for name, _ in obj_vars.items():
            if name.startswith('_'):
                continue
            if name in existing_instance_names or name in existing_class_names:
                continue

            # Create a property on the DataLoaderInterface class that forwards to
            # the underlying dataloader attribute. Using a property keeps the
            # attribute live (reads reflect dataloader changes).
            try:
                def _make_getter(n):
                    return lambda inst: getattr(inst.dataloader, n)

                getter = _make_getter(name)
                prop = property(fget=getter)
                setattr(self.__class__, name, prop)
            except Exception:
                # Best-effort: skip if we cannot set the property
                continue
        
        # Proxy class_names if available on the wrapped dataset
        if hasattr(self.dataset, 'class_names'):
             self.class_names = self.dataset.class_names

        # 2) Bind class-level callables (methods) of `obj` to this instance
        obj_cls_vars = getattr(obj.__class__, '__dict__', {})
        for name, member in obj_cls_vars.items():
            if name.startswith('_'):
                continue
            if name in existing_instance_names or name in existing_class_names:
                continue

            # Only consider callables defined on the class (functions/descriptors)
            if callable(member):
                try:
                    # getattr(obj, name) returns the bound method
                    bound = getattr(obj, name)
                    # Attach the bound method to the wrapper instance so that
                    # calling iface.name(...) calls obj.name(...)
                    setattr(self, name, bound)
                except Exception:
                    # If we cannot bind, skip gracefully
                    continue

    # -------------------------------------------------------------------------
    # Dataset-like helpers for trainer_services._dataset_to_df compatibility
    # -------------------------------------------------------------------------
    @property
    def wrapped_dataset(self):
        """
        For compatibility with code that expects dataset wrappers exposing
        `wrapped_dataset` (like in trainer_services._dataset_to_df).

        Prefer the tracking wrapper when available.
        """
        if hasattr(self, "tracked_dataset") and self.tracked_dataset is not None:
            return self.tracked_dataset
        return self.dataset

    def __getitem__(self, idx):
        """
        Allow treating the interface as a dataset for code paths that do
        `raw_ds[i]` (e.g. _dataset_to_df fallback path).
        """
        base = self.wrapped_dataset
        if hasattr(base, "__getitem__"):
            return base[idx]
        raise TypeError("Underlying dataset is not indexable")

    # -------------------------------------------------------------------------
    # Core iterator protocol
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of batches (delegates to the wrapped dataloader)."""
        return len(self.dataloader)

    def __iter__(self) -> Iterator:
        """Return a self-iterating wrapper that auto-resets on exhaustion.

        Returning ``self`` ensures ``__next__`` is used, which already
        handles StopIteration by recreating the underlying iterator. This
        makes ``for batch in dataloader_interface`` loop forever over epochs
        without the user having to call ``reset_iterator`` manually.
        """ 
        self._sync_batch_size_from_ledger()
        self._wait_if_paused()
        self._reset_iterator()  # Reset
        return self._iterator

    def __next__(self) -> Any:
        """Retrieve the next batch; used when iterating directly over the interface."""
        self._sync_batch_size_from_ledger()
        res = self._next_batch()
        self._wait_if_paused()
        return res

    # -------------------------------------------------------------------------
    # Ledger / pause helpers
    # -------------------------------------------------------------------------
    def _sync_batch_size_from_ledger(self) -> None:
        """Optionally sync batch size from global hyperparams ledger."""
        if self._ledger_name is None:
            return

        try:
            hp_name = resolve_hp_name()
            if hp_name is None:
                # no hyperparams; optionally use a default
                try:
                    self.set_batch_size(1)
                except RuntimeError:
                    pass
                return

            latest = get_hyperparams(hp_name)
            data_cfg = latest.get("data", {})
            if self._ledger_name in data_cfg:
                bs = data_cfg[self._ledger_name].get("batch_size", None)
                if bs is not None:
                    try:
                        self.set_batch_size(bs)
                    except RuntimeError:
                        # user-supplied dataloader: cannot change size, ignore
                        pass
            else:
                # No config for this dataloader -> optional default
                try:
                    self.set_batch_size(64)
                except RuntimeError:
                    pass
        except Exception:
            # Don't let ledger issues break basic iteration
            return

    def _wait_if_paused(self) -> None:
        """If the global pause controller is paused, wait until resumed."""
        try:
            pause_controller.wait_if_paused()
        except Exception:
            # Fail-open if pause controller is not available
            pass

    # -------------------------------------------------------------------------
    # Batch iteration helpers
    # -------------------------------------------------------------------------
    def _next_batch(self) -> Any:
        """Return the next batch from the dataloader.

        If the iterator is exhausted it is automatically reset and iteration
        resumes. With num_workers > 0, this properly cleans up worker processes
        before resetting the iterator.
        """
        try:
            if self._iterator is None:
                self._reset_iterator()
            # Generate batch
            batch = next(self._iterator)
            # Count yielded samples to support iteration state capture/restore
            self._samples_yielded += 1
            return batch
        except StopIteration:
            # End of epoch: reset iterator and try again (starting new epoch)
            logger.debug(f"Epoch complete ({self._samples_yielded} samples yielded), resetting iterator for new epoch")
            # Reset offset tracking for new epoch
            self._sample_offset = 0
            self._reset_iterator()
            
            # Try to get first batch from new epoch
            try:
                batch = next(self._iterator)
                self._samples_yielded += 1
                return batch
            except StopIteration:
                # Dataloader is empty or has no more data even after reset
                sampler = self._mutable_batch_sampler
                diagnostic_info = {
                    "dataloader_len": len(self.dataloader),
                    "sampler_type": type(sampler).__name__ if sampler else None,
                }
                if sampler and hasattr(sampler, 'offset'):
                    diagnostic_info["sampler_offset"] = sampler.offset
                if sampler and hasattr(sampler, 'batch_size'):
                    diagnostic_info["sampler_batch_size"] = sampler.batch_size
                if self.tracked_dataset:
                    try:
                        diagnostic_info["dataset_len"] = len(self.tracked_dataset)
                        if hasattr(self.tracked_dataset, '_get_df_view'):
                            df = self.tracked_dataset._get_df_view()
                            diagnostic_info["deny_listed_count"] = (df[SampleStatsEx.DISCARDED.value] == True).sum() if SampleStatsEx.DISCARDED.value in df.columns else 0
                    except Exception as e:
                        diagnostic_info["dataset_info_error"] = str(e)
                
                logger.warning(
                    f"Dataloader exhausted after reset. Diagnostic info: {diagnostic_info}"
                )
                raise
        except Exception as e:
            # Log unexpected errors to help diagnosis with multiprocessing
            logger.error(f"Error in _next_batch: {e}", exc_info=True)
            raise

    # def _execute_offset(self) -> None:
    #     """
    #         Execute sample offset if set, skipping samples as needed.
    #         This is a fallback mechanism for user-supplied dataloaders where
    #         we cannot use an OffsetSampler.

    #         TODO (GP):
    #         We can reproduce the random generation of samples by restoring RNG state, if during the previous checkpoints, batchsize changed dynamically and shuffle is True.
    #     """
    #     if self._sample_offset > 0:
    #         current_bs = self.get_batch_size()
    #         # Fast-forward the iterator by the offset amount
    #         while len(self._skipped) < self._sample_offset:
    #             try:
    #                 bs = 4 if self._sample_offset - len(self._skipped) >= 4 else self._sample_offset - len(self._skipped)  # Autoscale bs to sample offset
    #                 self.set_batch_size(bs)
    #                 self._skipped.extend(next(self._iterator)[1].detach().cpu().tolist())
    #                 logger.debug(f"Offset sampler: skipped {len(self._skipped)}/{self._sample_offset}")
    #             except StopIteration as e:
    #                 logger.debug(f"Offset sampler: reached end of iterator while skipping: {e}")
    #                 self._reset_iterator()  # Reset iterator and try again

    #         self.set_batch_size(current_bs)
    #         self._skipped = []
    #         self._sample_offset = 0

    def _reset_iterator(self) -> None:
        """Reset the internal iterator so `_next_batch()` starts from the beginning.
        
        For dataloaders with num_workers > 0, this explicitly cleans up the old iterator
        and its worker processes before creating a new one to avoid deadlocks or resource leaks.
        Also resets the sampler's offset to ensure we don't skip samples on new epochs.
        """
        import gc
        import time
        
        # Explicitly delete old iterator to allow worker processes to be cleaned up
        if hasattr(self, '_iterator') and self._iterator is not None:
            try:
                del self._iterator
                logger.debug("Deleted old iterator for cleanup")
            except Exception as e:
                logger.debug(f"Failed to delete old iterator: {e}")
        
        # Force garbage collection to ensure worker processes are terminated
        # This is especially important when num_workers > 0
        try:
            gc.collect()
        except Exception:
            pass
        
        # Reset sampler's offset for new epoch (important: prevents skipping samples on subsequent epochs)
        if hasattr(self, '_mutable_batch_sampler') and self._mutable_batch_sampler is not None:
            if hasattr(self._mutable_batch_sampler, 'offset'):
                old_offset = self._mutable_batch_sampler.offset
                self._mutable_batch_sampler.offset = 0
                if old_offset > 0:
                    logger.debug(f"Reset sampler offset from {old_offset} to 0")
        
        # Give worker processes time to fully terminate (especially important with num_workers > 0)
        # Short delay to avoid race conditions when spawning new workers
        if hasattr(self.dataloader, 'num_workers') and self.dataloader.num_workers > 0:
            time.sleep(0.01)  # 10ms delay for worker cleanup
        
        # Create new iterator
        self._iterator = iter(self.dataloader)
        logger.debug(f"Created new iterator (num_workers={getattr(self.dataloader, 'num_workers', 'unknown')}, sampler_len={len(self._mutable_batch_sampler) if self._mutable_batch_sampler else 'N/A'})")

    def reset_iterator(self) -> None:
        """Recreate the internal iterator (e.g., after restoring RNG state).

        Call this after restore_rng_state() to get a fresh shuffle with the
        restored RNG state:

            rng_state = capture_rng_state()
            batch1 = next(dataloader_interface)
            restore_rng_state(rng_state)
            dataloader_interface.reset_iterator()  # Create new iterator with restored RNG
            batch1_repeat = next(dataloader_interface)  # Same batches!
        """
        self._reset_iterator()

    # -------------------------------------------------------------------------
    # Iteration state capture/restore for deterministic resume
    # -------------------------------------------------------------------------
    def capture_iteration_state(self) -> dict:
        """Capture current iteration position for later restoration.

        Returns a serializable dict that can be stored with checkpoints and
        later supplied to `restore_iteration_state` to resume at the same batch
        boundary. Works with and without shuffling. When shuffling, ensure
        RNG state is also captured/restored before calling `restore_iteration_state`.
        """
        return {
            "samples_yielded": int(self._samples_yielded),
            "batch_size": self.batch_size or 1
        }

    def restore_iteration_state(self, state: dict) -> None:
        """Restore iteration position efficiently without reprocessing skipped data.

        For dataloaders we built (with _dl_build_kwargs), this recreates the
        dataloader with an OffsetSampler that skips samples at the index level,
        avoiding expensive data loading and transforms for skipped batches.

        For shuffled loaders, call this after restoring RNG state.
        """
        try:
            samples_yielded = int(state.get("samples_yielded", 0))
            batch_size = int(state.get("batch_size", self.batch_size or 1))
        except Exception:
            samples_yielded = 0
            batch_size = self.batch_size or 1

        # Calculate sample offset (how many individual samples to skip)
        sample_offset = samples_yielded

        # If we own the dataloader construction, rebuild with offset sampler
        if getattr(self, "_dl_build_kwargs", None) is not None and sample_offset > 0:
            try:
                kwargs = dict(self._dl_build_kwargs)
                # Remove kwargs that conflict with using batch_sampler
                kwargs.pop("batch_size", None)
                shuffle = kwargs.pop("shuffle", False)
                num_workers = kwargs.pop("num_workers", 0)
                drop_last = kwargs.pop("drop_last", False)
                pin_memory = kwargs.pop("pin_memory", False)
                collate_fn = kwargs.pop("collate_fn", None)
                kwargs.pop("sampler", None)
                kwargs.pop("drop_last", None)
                kwargs.pop("shuffle", None)

                # Create a new sampler with the offset and batch size, and rebuild the dataloader
                sampler = WeightsLabDataSampler(
                    self.tracked_dataset,
                    tracked_dataset=self.tracked_dataset,
                    shuffle=shuffle,
                    offset=sample_offset,
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
                self._mutable_batch_sampler = sampler
                self._sample_offset = 0

                # Rebuild dataloader with offset sampler
                self.dataloader = DataLoader(
                    self.tracked_dataset,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=collate_fn,
                    # Ensure no conflicting args are passed alongside batch_sampler
                    **filter_kwargs_for_callable(DataLoader, kwargs)
                )

                # Reset iterator and counter
                self._iterator = None
                self._samples_yielded = samples_yielded
                return
            except Exception as e:
                logger.warning(f"Failed to restore with offset sampler, falling back to fast-forward: {e}")

        # Fallback: fast-forward approach (less efficient but works for user-supplied dataloaders)
        self._reset_iterator()
        for _ in range(max(0, samples_yielded)):
            try:
                next(self._iterator)
                self._samples_yielded += 1
            except StopIteration:
                break

    # -------------------------------------------------------------------------
    # Batch-size management
    # -------------------------------------------------------------------------
    def set_batch_size(self, new_batch_size: int) -> None:
        """Change the effective batch size used by this interface.

        If we own a mutable batch sampler, update its `batch_size` in-place.
        Otherwise, if we created the DataLoader and kept build kwargs,
        recreate it with the new batch size.

        If a user-supplied DataLoader was wrapped, this operation is not
        supported and will raise `RuntimeError`.
        """
        new_batch_size = int(new_batch_size)

        # If effective batch size is unchanged, do nothing
        if self.batch_size is not None and self.batch_size == new_batch_size:
            return

        # Case 1: we have a mutable batch sampler
        if getattr(self, "_mutable_batch_sampler", None) is not None:
            self._mutable_batch_sampler.batch_size = new_batch_size
            self._reset_iterator()
            return

        # Case 2: we created the dataloader and stored build kwargs
        if getattr(self, "_dl_build_kwargs", None) is not None:
            try:
                self._dl_build_kwargs["batch_size"] = new_batch_size
                kwargs = dict(self._dl_build_kwargs)

                batch_size = kwargs.pop("batch_size", None)
                shuffle = kwargs.pop("shuffle", False)
                num_workers = kwargs.pop("num_workers", 0)
                drop_last = kwargs.pop("drop_last", False)
                pin_memory = kwargs.pop("pin_memory", False)
                collate_fn = kwargs.pop("collate_fn", None)

                # Rebuild sampler & dataloader if we had one
                if getattr(self, "_mutable_batch_sampler", None) is not None:
                    sampler = WeightsLabDataSampler(
                        self.tracked_dataset,
                        tracked_dataset=self.tracked_dataset,
                        shuffle=shuffle,
                        offset=0,
                        batch_size=batch_size,
                        drop_last=drop_last,
                    )
                    self._mutable_batch_sampler = sampler
                    self.dataloader = DataLoader(
                        self.tracked_dataset,
                        batch_sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn,
                        **kwargs,
                    )
                else:
                    # Plain DataLoader with batch_size=...
                    self.dataloader = DataLoader(
                        self.tracked_dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn,
                        **kwargs,
                    )

                self._reset_iterator()
                return
            except Exception as e:
                raise RuntimeError(f"Failed to update batch size: {e}") from e

        # Case 3: user-supplied dataloader, no build kwargs -> cannot change
        raise RuntimeError("Cannot change batch size for a user-supplied DataLoader")

    def get_batch_size(self) -> Optional[int]:
        """Return the current effective batch size or None if unknown."""
        # Prefer mutable sampler if present
        try:
            if getattr(self, "_mutable_batch_sampler", None) is not None:
                bs = getattr(self._mutable_batch_sampler, "batch_size", None)
                if bs is not None:
                    return int(bs)
        except Exception:
            pass

        # Common DataLoader attribute when built with `batch_size=`
        try:
            bs = getattr(self.dataloader, "batch_size", None)
            if bs is not None:
                return int(bs)
        except Exception:
            pass

        # If built with a batch_sampler, try to inspect it
        try:
            batch_sampler = getattr(self.dataloader, "batch_sampler", None)
            bs2 = getattr(batch_sampler, "batch_size", None)
            if bs2 is not None:
                return int(bs2)
        except Exception:
            pass

        return None

    @property
    def batch_size(self) -> Optional[int]:
        """Property exposing the current effective batch size."""
        return self.get_batch_size()

    # -------------------------------------------------------------------------
    # Dataset helpers
    # -------------------------------------------------------------------------
    def as_records(self, limit: int = -1):
        """Return dataset records via the underlying `as_records()`.

        We try `tracked_dataset.as_records()` first, then fall back to
        `dataset.as_records()` if present.
        """
        if hasattr(self.tracked_dataset, "as_records"):
            return self.tracked_dataset.as_records(limit)
        if hasattr(self.dataset, "as_records"):
            return self.dataset.as_records(limit)
        raise AttributeError("Wrapped dataset does not implement 'as_records()'")

    def get_dataloader(self) -> DataLoader:
        """Return the underlying `torch.utils.data.DataLoader`."""
        return self.dataloader

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DataLoaderInterface(dataset="
            f"{getattr(self.dataset, '__class__', type(self.dataset))}, "
            f"batch_size={self.batch_size})"
        )
