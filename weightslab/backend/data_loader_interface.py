"""Backend: Data loader interface for wrapping datasets and dataloaders.

This module provides a lightweight interface around PyTorch datasets and
`torch.utils.data.DataLoader` instances used by WeightsLab. It is intentionally
small: it normalizes inputs (dataset or DataLoader), exposes convenience
methods used by the rest of the codebase (like `as_records`) and provides a
resettable iterator and a safe `next_batch()` helper.
"""

from typing import Any, Iterator, Optional

import torch as th
from torch.utils.data import DataLoader, Dataset
from weightslab.data.data_samples_with_ops import \
    DataSampleTrackingWrapper
from weightslab.ledgers import register_dataloader


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
                self.reset_iterator)
        else:
            self.tracked_dataset = DataSampleTrackingWrapper(
                data_loader_or_dataset
            )  # Track the dataset
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
        self.is_training = is_training

        # Internal iterator used by `next_batch`
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

    def __iter__(self) -> Iterator:
        """Return an iterator over batches (delegates to the wrapped dataloader)."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Return the number of batches (delegates to the wrapped dataloader)."""
        return len(self.dataloader)
    
    def __next__(self) -> Any:
        return self.next_batch()

    def next_batch(self) -> Any:
        """Return the next batch from the dataloader. If the iterator is
        exhausted it is automatically reset and iteration resumes.
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            if not self.is_training:
                raise StopIteration("End of dataloader reached.")
            self.reset_iterator()
            batch = next(self._iterator)
        # Record last batch in thread-local store so other components (model/loss wrappers)
        # can access it for automatic dataset-statistics updates when inside a guard.
        return batch

    def reset_iterator(self) -> None:
        """Reset the internal iterator so `next_batch()` starts from the
        beginning.
        """
        self._iterator = iter(self.dataloader)
    
    def as_records(self, limit: int = -1):
        """Return dataset records via the tracked dataset's `as_records()`
        method if available.

        Args:
            limit: optional limit on records passed to underlying implementation.
        """
        if hasattr(self.tracked_dataset, "as_records"):
            return self.tracked_dataset.as_records(limit)
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
            self.reset_iterator()
            return
        raise AttributeError(
            "Wrapped dataset does not support setting a 'transform' attribute"
        )

    def get_dataloader(self) -> DataLoader:
        """Return the underlying `torch.utils.data.DataLoader`."""
        return self.dataloader

    def __repr__(self) -> str:
        return (
            f"DataLoaderInterface(dataset={getattr(self.dataset, '__class__', type(self.dataset))}, "
            f"batch_size={getattr(self.dataloader, 'batch_size', None)})"
        )


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

    wrapper = DataLoaderInterface(train_dataset, batch_size=8, shuffle=True)
    batch = wrapper.next_batch()
    print("Got batch with", len(batch), "elements")
