import numpy as np

from torch.utils import data
from pathlib import Path


class BaseDataset(data.Dataset):
    def __init__(
        self,
        transform_args,
        base_dir,
        transform,
    ):
        super().__init__()
        self.transform_args = transform_args
        self._base_dir = Path(base_dir)
        self.images = []
        self.transform = transform

    def __len__(self):
        return len(self.images)


def lbl_contains_any(lbl, lbl_list):
    unseen_pixel_mask = np.in1d(lbl.ravel(), lbl_list)
    if np.sum(unseen_pixel_mask) > 0:
        return True
    return False


def lbl_contains_all(lbl, lbl_list):
    unseen_pixel_mask = np.in1d(lbl.ravel(), lbl_list)
    if np.prod(unseen_pixel_mask) > 0:
        return True
    return False
