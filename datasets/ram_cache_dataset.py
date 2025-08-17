from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from .base_dataset import (
    BaseImageDataset,
    validate_compose_transform,
    TransformType
)
from .utils import _to_hwc, _to_chw

class CompactRAMCache(Dataset):
    def __init__(
        self,
        dataset: BaseImageDataset,
        cache_size=None,
        cache_dtype=np.float32,
        transform=None,
        input_only_transform=None,
        target_only_transform=None,
        max_ram_bytes=None
    ):
        self._ds = dataset
        n = len(dataset)
        if cache_size is None:
            cache_size = n
        cache_size = min(cache_size, n)

        # Probe one item for shape
        x0, y0 = dataset._get_raw_item(0)
        Cx, Hx, Wx = x0.shape
        Cy, Hy, Wy = y0.shape

        # Compute per-item bytes with requested dtype
        dtype = np.dtype(cache_dtype)
        item_bytes = (x0.size + y0.size) * dtype.itemsize

        if max_ram_bytes is not None:
            cache_size = max(1, min(cache_size, max_ram_bytes // item_bytes))

        self._cache_size = cache_size
        self._cache_dtype = dtype

        # Preallocate contiguous blocks
        self._inputs  = np.empty((cache_size, Cx, Hx, Wx), dtype=dtype)
        self._targets = np.empty((cache_size, Cy, Hy, Wy), dtype=dtype)

        # Fill cache (first cache_size items)

        for i in tqdm(range(cache_size), desc="Caching items", total=cache_size):
            xi, yi = dataset._get_raw_item(i)
            # Cast without extra copies if already matching dtype
            self._inputs[i]  = np.asarray(xi, dtype=dtype)
            self._targets[i] = np.asarray(yi, dtype=dtype)

        self.transform = transform
        self.input_only_transform = input_only_transform
        self.target_only_transform = target_only_transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        if idx < self._cache_size:
            x = self._inputs[idx]
            y = self._targets[idx]
        else:
            # fallback to backing dataset beyond cache window
            x, y = self._ds._get_raw_item(idx)
            x = np.asarray(x, dtype=self._cache_dtype)
            y = np.asarray(y, dtype=self._cache_dtype)

        x, y = self._apply_transform(x, y)

        # Return torch tensors (convert once here, not in cache)
        return (
            torch.from_numpy(x).to(torch.float32), 
            torch.from_numpy(y).to(torch.float32)
        )
    
    @property
    def transform(self) -> Optional[TransformType]:
        """
        Returns the transform applied to both input and target images.
        """
        return self._transform
    @transform.setter
    def transform(self, value: Optional[TransformType]=None):
        """
        Sets the transform to be applied to both input and target images.
        """
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=True)
        self._transform = value

    @property
    def input_only_transform(self) -> Optional[TransformType]:
        """
        Returns the input-only transform.
        """
        return self._input_only_transform
    @input_only_transform.setter
    def input_only_transform(self, value: Optional[TransformType]=None):
        """
        Sets the input-only transform.
        """
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=False)
        self._input_only_transform = value

    @property
    def target_only_transform(self) -> Optional[TransformType]:
        """
        Returns the target-only transform.
        """
        return self._target_only_transform
    @target_only_transform.setter
    def target_only_transform(self, value: Optional[TransformType]=None):
        """
        Sets the target-only transform.
        """
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=False)
        self._target_only_transform = value

    def _apply_transform(
        self,
        input_image: np.ndarray,
        target_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:        
        """
        Helper method to apply the dataset's configured image transformation
        to an input and target image stack pair.
        
        :param input_image: Input image (C, H, W) as a NumPy array.
        :param target_image: Target image (C, H, W) as a NumPy array.
        :return: Tuple of transformed input and target images.
        """

        inp_ref, tgt_ref = input_image, target_image

        # Albumentations expects HWC
        inp_hwc = _to_hwc(input_image)
        tgt_hwc = _to_hwc(target_image)

        # First apply shared transforms
        if self.transform is not None:
            t = self.transform(image=inp_hwc, target=tgt_hwc)
            inp_hwc = t['image']
            tgt_hwc = t['target']
        
        # Then apply input-only and target-only transforms
        if self.input_only_transform is not None:
            inp_hwc = self.input_only_transform(image=inp_hwc)['image']

        if self.target_only_transform is not None:
            tgt_hwc = self.target_only_transform(image=tgt_hwc)['image']

        # Convert back to CHW (match original shapes)
        inp_chw = _to_chw(inp_hwc, ref=inp_ref)
        tgt_chw = _to_chw(tgt_hwc, ref=tgt_ref)
        
        return inp_chw, tgt_chw
