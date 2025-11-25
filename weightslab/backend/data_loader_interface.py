"""
Backend: Data loader interface for wrapping datasets and dataloaders.

This module provides a lightweight interface around PyTorch datasets and
`torch.utils.data.DataLoader` instances used by WeightsLab. It is intentionally
small: it normalizes inputs (dataset or DataLoader), exposes convenience
methods used by the rest of the codebase (like `as_records`) and provides a
resettable iterator and a safe `next_batch()` helper.
"""

import torch as th

from typing import Any, Iterator, Optional      
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler

from weightslab.components.global_monitoring import \
    pause_controller
from weightslab.data.data_samples_with_ops import \
    DataSampleTrackingWrapper
from weightslab.ledgers import register_dataloader, get_hyperparams, list_hyperparams


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
    """

    def __init__(
        self,
        data_loader_or_dataset: Any,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        collate_fn: Optional[Any] = None,
        name: Optional[str] = None,
        register: bool = True,
        weak: bool = False,
        is_training: bool = False,
        # Note: the interface now always uses a mutable batch sampler when
        # building a DataLoader from a Dataset. This allows changing the
        # effective batch size at runtime via `set_batch_size()`.
        **kwargs,
    ) -> None:
        
        # Normalize inputs
        self.dataset: Dataset | DataLoader = data_loader_or_dataset
        if isinstance(data_loader_or_dataset, DataLoader):
            self.dataloader: DataLoader = data_loader_or_dataset
            self.tracked_dataset = DataSampleTrackingWrapper(
                self.dataloader
            )  # Track the dataset
            self.tracked_dataset._map_updates_hook_fns.append(
                self._reset_iterator)
        else:
            self.tracked_dataset = DataSampleTrackingWrapper(
                data_loader_or_dataset
            )  # Track the dataset
            # store kwargs so we can recreate dataloader if needed
            self._dl_build_kwargs = dict(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
            self._dl_build_kwargs.update(kwargs or {})

            # Always use a MutableBatchSampler when we build the DataLoader
            # from a Dataset. This enables safe runtime updates of batch size.
            # Choose base sampler according to shuffle flag
            base_sampler = RandomSampler(self.tracked_dataset) if shuffle else SequentialSampler(self.tracked_dataset)
            # Lazy define the sampler class here to avoid exposing it at module level
            class MutableBatchSampler:
                """A simple mutable batch sampler that yields lists of indices

                Changing the `batch_size` attribute at runtime will affect
                subsequent iterations.
                """
                def __init__(self, base_sampler, batch_size, drop_last=False):
                    self.base_sampler = base_sampler
                    self.batch_size = int(batch_size)
                    self.drop_last = bool(drop_last)

                def __iter__(self):
                    batch = []
                    for idx in self.base_sampler:
                        batch.append(idx)
                        if len(batch) >= int(self.batch_size):
                            yield list(batch)
                            batch = []
                    if batch and not self.drop_last:
                        yield list(batch)

                def __len__(self):
                    try:
                        total = len(self.base_sampler)
                        b = max(1, int(self.batch_size))
                        return (total + b - 1) // b
                    except Exception:
                        raise TypeError("len not supported for this sampler")

            mbs = MutableBatchSampler(base_sampler, batch_size, drop_last=drop_last)
            self._mutable_batch_sampler = mbs
            # Construct dataloader using our batch_sampler
            self.dataloader = DataLoader(
                self.tracked_dataset,
                batch_sampler=mbs,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                **kwargs,
            )
        self.is_training = is_training

        # Internal iterator used by `_next_batch`
        self._iterator: Iterator = iter(self.dataloader)

        # Optionally register in the global ledger for cross-thread access.
        # If no explicit `name` is provided, try to infer a friendly name from
        # the wrapped dataset class name; otherwise fall back to '_dataloader'.
        if register:
            reg_name = name or \
                getattr(self.dataset, "__name__", None) or \
                getattr(
                    self.dataset,
                    "__class__",
                    type(self.dataset)
                ).__name__ or "_dataloader"
            self._ledger_name = reg_name
            try:
                register_dataloader(reg_name, self, weak=weak)
            except Exception:
                pass

    def __len__(self) -> int:
        """Return the number of batches (delegates to the wrapped dataloader)."""
        return len(self.dataloader)
    
    def __iter__(self) -> Iterator:
        """Return an iterator over batches (delegates to the wrapped dataloader)."""
        if self._ledger_name in get_hyperparams(list_hyperparams()[0])['data']:
            bs = get_hyperparams(list_hyperparams()[0])['data'][self._ledger_name]['batch_size']
            self.set_batch_size(bs)  # check and update batch size before iterating
        self._wait_if_paused()
        return iter(self.dataloader)
    
    def __next__(self) -> Any:
        if self._ledger_name in get_hyperparams(list_hyperparams()[0])['data']:
            bs = get_hyperparams(list_hyperparams()[0])['data'][self._ledger_name]['batch_size']
            self.set_batch_size(bs)  # check and update batch size before iterating
        self._wait_if_paused()
        return self._next_batch()

    def __repr__(self) -> str:
        return (
            f"DataLoaderInterface(dataset={getattr(self.dataset, '__class__', type(self.dataset))}, "
            f"batch_size={getattr(self.dataloader, 'batch_size', None)})"
        )
    
    def _wait_if_paused(self):
        """
        If the global pause controller is paused, wait until resumed.
        """
        pause_controller.wait_if_paused()

    def _next_batch(self) -> Any:
        """Return the next batch from the dataloader. If the iterator is
        exhausted it is automatically reset and iteration resumes.
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            if not self.is_training:
                raise StopIteration("End of dataloader reached.")
            self._reset_iterator()
            batch = next(self._iterator)
        # Record last batch in thread-local store so other components (model/loss wrappers)
        # can access it for automatic dataset-statistics updates when inside a guard.
        return batch

    def _reset_iterator(self) -> None:
        """Reset the internal iterator so `_next_batch()` starts from the
        beginning.
        """
        self._iterator = iter(self.dataloader)

    def set_batch_size(self, new_batch_size: int) -> None:
        """Change the effective batch size used by this interface.

        If the dataloader was created with `mutable_batch_sampler=True` this
        will update the sampler's `batch_size` in-place and reset the
        iterator. If the dataloader was built normally and the interface
        created the underlying DataLoader, we will recreate the DataLoader
        with the new batch size. If a user-supplied DataLoader was wrapped
        (i.e. the interface received a `DataLoader` instance) this operation
        is not supported and will raise `RuntimeError`.
        """
        if hasattr(self, 'batch_size') and self.batch_size == int(new_batch_size):
            return  # no change needed
        
        if hasattr(self, '_mutable_batch_sampler') and self._mutable_batch_sampler is not None:
            try:
                self._mutable_batch_sampler.batch_size = int(new_batch_size)
                self._reset_iterator()
                return
            except Exception:
                raise

        # If we created the dataloader ourselves, recreate it with new size
        if isinstance(self.dataset, Dataset) and not isinstance(self.dataloader, DataLoader) or getattr(self, '_dl_build_kwargs', None) is not None:
            try:
                # update stored kwargs and rebuild
                self._dl_build_kwargs['batch_size'] = int(new_batch_size)
                kwargs = dict(self._dl_build_kwargs)
                # preserve shuffle flag in kwargs; decide whether to use batch_sampler
                batch_size = kwargs.pop('batch_size', None)
                shuffle = kwargs.pop('shuffle', False)
                num_workers = kwargs.pop('num_workers', 0)
                drop_last = kwargs.pop('drop_last', False)
                pin_memory = kwargs.pop('pin_memory', False)
                collate_fn = kwargs.pop('collate_fn', None)

                # If we had a mutable sampler before, rebuild with MutableBatchSampler
                if getattr(self, '_mutable_batch_sampler', None) is not None:
                    base_sampler = RandomSampler(self.tracked_dataset) if shuffle else SequentialSampler(self.tracked_dataset)
                    mbs = type(self._mutable_batch_sampler)(base_sampler, batch_size, drop_last=drop_last)
                    self._mutable_batch_sampler = mbs
                    self.dataloader = DataLoader(self.tracked_dataset, batch_sampler=mbs, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, **kwargs)
                else:
                    self.dataloader = DataLoader(self.tracked_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory, collate_fn=collate_fn, **kwargs)

                self._reset_iterator()
                return
            except Exception as e:
                raise RuntimeError(f"Failed to update batch size: {e}")

        # If we reach here, we wrapped a user-supplied DataLoader and can't change it
        raise RuntimeError("Cannot change batch size for a user-supplied DataLoader")

    def get_batch_size(self) -> Optional[int]:
        """Return the current effective batch size or None if unknown.

        This inspects mutable samplers first, then common DataLoader
        attributes (`batch_size`, `batch_sampler.batch_size`).
        """
        try:
            if getattr(self, '_mutable_batch_sampler', None) is not None:
                return int(getattr(self._mutable_batch_sampler, 'batch_size', None))
        except Exception:
            pass

        # Common DataLoader attribute when built with `batch_size=`
        try:
            bs = getattr(self.dataloader, 'batch_size', None)
            if bs is not None:
                return int(bs)
        except Exception:
            pass

        # If built with a batch_sampler, try to inspect it
        try:
            bs2 = getattr(getattr(self.dataloader, 'batch_sampler', None), 'batch_size', None)
            if bs2 is not None:
                return int(bs2)
        except Exception:
            pass

        return None

    @property
    def batch_size(self) -> Optional[int]:
        return self.get_batch_size()
    
    def as_records(self, limit: int = -1):
        """Return dataset records via the underlying dataset's `as_records()`
        method if available.

        Args:
            limit: optional limit on records passed to underlying implementation.
        """
        if hasattr(self.dataset, "as_records"):
            return self.dataset.as_records(limit)
        raise AttributeError(
            "Wrapped dataset does not implement 'as_records()'"
        )

    def set_transform(self, transform: Any) -> None:
        """Set a `transform` attribute on the wrapped dataset when supported.

        Many torchvision datasets expose a `transform` attribute. This helper
        allows swapping it at runtime.
        """
        if hasattr(self.dataset, "transform"):
            setattr(self.dataset, "transform", transform)
            # In case the DataLoader had pinned memory or similar, recreate
            # the iterator to ensure consistent behavior.
            self._reset_iterator()
            return
        raise AttributeError(
            "Wrapped dataset does not support setting a 'transform' attribute"
        )

    def get_dataloader(self) -> DataLoader:
        """Return the underlying `torch.utils.data.DataLoader`."""
        return self.dataloader


if __name__ == "__main__":
    # Quick demo when running this module directly. This mirrors the
    # previous example but only demonstrates the interface usage.
    import os
    import tempfile
    from torchvision import datasets, transforms

    TMP_DIR = tempfile.mkdtemp()

    train_dataset = datasets.FashionMNIST(
        root=os.path.join(TMP_DIR, "data"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Demonstrate mutable batch sampler usage
    wrapper = DataLoaderInterface(train_dataset, batch_size=8, shuffle=True, mutable_batch_sampler=True)
    print("Initial effective batch_size:", wrapper.get_batch_size())
    batch = wrapper._next_batch()
    # depending on collate_fn/tensor structure, len(batch) may reflect batch size
    try:
        print("Got batch with", len(batch), "elements")
    except Exception:
        print("Got a batch (unable to determine length)")

    # Change batch size at runtime
    wrapper.set_batch_size(16)
    print("After set_batch_size(16), effective batch_size:", wrapper.batch_size)
    # fetch another batch (uses new batch size for subsequent iterations)
    batch2 = wrapper._next_batch()
    try:
        print("Got batch with", len(batch2), "elements")
    except Exception:
        print("Got a batch (unable to determine length)")
