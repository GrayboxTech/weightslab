# =============================================================================
# MNIST dataset yielding per-sample uids for the shared WeightsLab ledger
# =============================================================================
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTIdx(Dataset):
    """Yields (image, uid, label). uid namespaced by split so train/test don't
    collide in the shared ledger; fast_get_label skips decode at ledger init."""
    def __init__(self, root, train, base):
        self.m = datasets.MNIST(root, train=train, download=True, transform=None)
        self.t = transforms.ToTensor(); self.base = base

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        img, lab = self.m[i]
        return self.t(img), self.base + i, lab

    def fast_get_label(self, i):
        return int(self.m.targets[i])
