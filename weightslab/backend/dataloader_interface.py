"""
Backend: Data loader interface for wrapping datasets and dataloaders.

This module provides a lightweight interface around PyTorch datasets and
`torch.utils.data.DataLoader` instances used by WeightsLab. It is intentionally
small: it normalizes inputs (dataset or DataLoader), exposes convenience
methods used by the rest of the codebase (like `as_records`) and provides a
resettable iterator and a safe `next_batch()` helper.

It also supports:
- a mutable batch sampler for runtime batch-size changes
- global pause control
- registration in a global ledger and dynamic batch-size updates based on
  hyperparameters
"""

from typing import Any, Iterator, Optional

import torch as th  # noqa: F401
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler

from weightslab.components.global_monitoring import pause_controller
from weightslab.data.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.backend.ledgers import (
    register_dataloader,
    get_hyperparams,
    list_hyperparams,
)


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
        pin_memory: bool = False,
        collate_fn: Optional[Any] = None,
        name: Optional[str] = None,
        register: bool = True,
        weak: bool = False,
        is_training: bool = False,
        # kept for backwards compatibility; always use mutable sampler when we
        # build the DataLoader from a Dataset, but do not pass this to PyTorch.
        mutable_batch_sampler: bool = True,  # noqa: F841
        **kwargs,
    ) -> None:
        # Strip out our own kwargs so they don't get passed to DataLoader
        kwargs = dict(kwargs)
        kwargs.pop("mutable_batch_sampler", None)

        # Normalize inputs
        self.dataset: Dataset | DataLoader = data_loader_or_dataset

        # Internal flags / helpers
        self._mutable_batch_sampler = None
        self._dl_build_kwargs: Optional[dict] = None

        if isinstance(data_loader_or_dataset, DataLoader):
            # User-supplied dataloader
            self.dataloader: DataLoader = data_loader_or_dataset
            self.tracked_dataset = DataSampleTrackingWrapper(self.dataloader)
            self.tracked_dataset._map_updates_hook_fns.append(
                self._reset_iterator
            )
        else:
            # Dataset supplied: wrap and build our own DataLoader with a mutable batch sampler
            self.tracked_dataset = DataSampleTrackingWrapper(data_loader_or_dataset)

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

            # Choose base sampler according to shuffle flag
            base_sampler = (
                RandomSampler(self.tracked_dataset)
                if shuffle
                else SequentialSampler(self.tracked_dataset)
            )

            # Inner class: mutable batch sampler
            class MutableBatchSampler:
                """A simple mutable batch sampler that yields lists of indices.

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

            mbs = MutableBatchSampler(
                base_sampler, batch_size=batch_size, drop_last=drop_last
            )
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

        self.init_attributes(self.dataloader)

        self.is_training = is_training

        # Internal iterator used by `_next_batch` / `next_batch`
        self._iterator: Iterator = iter(self.dataloader)

        # Optionally register in the global ledger for cross-thread access.
        # If no explicit `name` is provided, try to infer a friendly name from
        # the wrapped dataset class name; otherwise fall back to '_dataloader'.
        self._ledger_name = None
        if register:
            reg_name = (
                name
                or getattr(self.dataset, "__name__", None)
                or getattr(self.dataset, "__class__", type(self.dataset)).__name__
                or "_dataloader"
            )
            self._ledger_name = reg_name
            try:
                register_dataloader(reg_name, self, weak=weak)
            except Exception:
                # Best-effort: ignore registration failures
                pass

    def init_attributes(self, obj):
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
        for name, value in obj_vars.items():
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
        """Return an iterator over batches (delegates to the wrapped dataloader)."""
        self._sync_batch_size_from_ledger()
        self._wait_if_paused()
        return iter(self.dataloader)

    def __next__(self) -> Any:
        """Retrieve the next batch; used when iterating directly over the interface."""
        self._sync_batch_size_from_ledger()
        self._wait_if_paused()
        return self._next_batch()

    # -------------------------------------------------------------------------
    # Ledger / pause helpers
    # -------------------------------------------------------------------------
    def _sync_batch_size_from_ledger(self) -> None:
        """Optionally sync batch size from global hyperparams ledger."""
        if self._ledger_name is None:
            return

        try:
            params = list_hyperparams()
            if not params:
                # no hyperparams; optionally use a default
                try:
                    self.set_batch_size(64)
                except RuntimeError:
                    pass
                return

            latest = get_hyperparams(params[0])
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
        resumes (unless `is_training=False`, in which case StopIteration is
        propagated).
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            if not self.is_training:
                raise StopIteration("End of dataloader reached.")
            self._reset_iterator()
            batch = next(self._iterator)
        return batch

    def _reset_iterator(self) -> None:
        """Reset the internal iterator so `_next_batch()` starts from the beginning."""
        self._iterator = iter(self.dataloader)

    # Backwards-compatible public names
    def next_batch(self) -> Any:
        """Backwards-compatible alias for `_next_batch()`."""
        return self._next_batch()

    def reset_iterator(self) -> None:
        """Backwards-compatible alias for `_reset_iterator()`."""
        return self._reset_iterator()

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

                # Rebuild base sampler & mutable sampler if we had one
                if getattr(self, "_mutable_batch_sampler", None) is not None:
                    base_sampler = (
                        RandomSampler(self.tracked_dataset)
                        if shuffle
                        else SequentialSampler(self.tracked_dataset)
                    )
                    mbs_cls = type(self._mutable_batch_sampler)
                    mbs = mbs_cls(
                        base_sampler, batch_size=batch_size, drop_last=drop_last
                    )
                    self._mutable_batch_sampler = mbs
                    self.dataloader = DataLoader(
                        self.tracked_dataset,
                        batch_sampler=mbs,
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

    def set_transform(self, transform: Any) -> None:
        """Set a `transform` attribute on the wrapped dataset when supported.

        Many torchvision datasets expose a `transform` attribute. This helper
        allows swapping it at runtime.
        """
        if hasattr(self.dataset, "transform"):
            setattr(self.dataset, "transform", transform)
            self._reset_iterator()
            return
        raise AttributeError(
            "Wrapped dataset does not support setting a 'transform' attribute"
        )

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


if __name__ == "__main__":
    # Quick demo when running this module directly.
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
    wrapper = DataLoaderInterface(
        train_dataset,
        batch_size=8,
        shuffle=True,
        mutable_batch_sampler=True,  # accepted but not passed to DataLoader
    )
    print("Initial effective batch_size:", wrapper.get_batch_size())
    batch = wrapper.next_batch()
    try:
        print("Got batch with", len(batch), "elements")
    except Exception:
        print("Got a batch (unable to determine length)")

    # Change batch size at runtime
    wrapper.set_batch_size(16)
    print("After set_batch_size(16), effective batch_size:", wrapper.batch_size)
    batch2 = wrapper.next_batch()
    try:
        print("Got batch with", len(batch2), "elements")
    except Exception:
        print("Got a batch (unable to determine length)")
