import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class CompactRAMCache(Dataset):
    def __init__(
        self,
        dataset,
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

        # Apply transforms lazily (don’t cache transformed copies)
        if self.transform is not None:
            t = self.transform(image=x, target=y)
            x, y = t["image"], t["target"]
        if self.input_only_transform is not None:
            x = self.input_only_transform(image=x)["image"]
        if self.target_only_transform is not None:
            y = self.target_only_transform(image=y)["image"]

        # Return torch tensors (convert once here, not in cache)
        return torch.from_numpy(x).to(torch.float32), torch.from_numpy(y).to(torch.float32)
