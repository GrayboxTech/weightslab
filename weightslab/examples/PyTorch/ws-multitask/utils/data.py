"""
MNIST multi-task dataset: each sample has a digit class label (classification)
and a tight bounding box of the non-zero pixels (localization).

Target format follows the WeightsLab detection convention:
    tensor of shape [N, 6] with columns [x1, y1, x2, y2, class_id, confidence]
    all coordinates normalized to [0, 1].

This lets the WeightsLab UI render ground-truth bboxes over each sample.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTMultiTaskDataset(Dataset):
    """MNIST with per-sample tight bounding boxes synthesized from pixel intensity."""

    def __init__(self, root, train=True, download=True, transform=None, max_samples=None):
        try:
            self._mnist = datasets.MNIST(root=root, train=train, download=download, transform=None)
        except RuntimeError:
            self._mnist = datasets.MNIST(root=root, train=train, download=True, transform=None)

        self.transform = transform or transforms.ToTensor()
        self.max_samples = max_samples
        self._length = min(len(self._mnist), max_samples) if max_samples else len(self._mnist)

    def __len__(self):
        return self._length

    def _compute_bbox(self, img_tensor):
        """Return (x1, y1, x2, y2) normalized to [0,1] for the digit's tight bbox."""
        mask = img_tensor.squeeze(0) > 0.1
        if not mask.any():
            return 0.0, 0.0, 1.0, 1.0

        rows_with_signal = mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols_with_signal = mask.any(dim=0).nonzero(as_tuple=True)[0]

        H, W = img_tensor.shape[-2], img_tensor.shape[-1]
        y1 = float(rows_with_signal.min()) / H
        y2 = float(rows_with_signal.max()) / H
        x1 = float(cols_with_signal.min()) / W
        x2 = float(cols_with_signal.max()) / W
        return x1, y1, x2, y2

    def __getitem__(self, idx):
        """Returns (image, idx, target) where target is a [1, 6] detection tensor."""
        image, label = self._mnist[idx]
        image = self.transform(image)
        x1, y1, x2, y2 = self._compute_bbox(image)
        target = torch.tensor(
            [[x1, y1, x2, y2, float(label), 1.0]], dtype=torch.float32
        )
        return image, idx, target


def multitask_collate(batch):
    """Collate for detection-format targets: targets remains a list of [N,6] tensors."""
    images, ids, targets = zip(*batch)
    return torch.stack(images), torch.tensor(ids, dtype=torch.long), list(targets), {}
