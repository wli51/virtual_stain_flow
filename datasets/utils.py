"""
datasets/utils.py
"""

import numpy as np

def _to_hwc(x: np.ndarray) -> np.ndarray:
    # (C,H,W) -> (H,W,C) ; (C,H,W,K) -> (H,W,K*C) by flattening channel dims
    if x.ndim == 3:
        return np.moveaxis(x, 0, -1)
    elif x.ndim == 4:
        c, h, w, k = x.shape
        x = x.transpose(1, 2, 0, 3).reshape(h, w, c * k)
        return x
    else:
        raise ValueError(f"Unexpected ndim {x.ndim} for CHW-like tensor")

def _to_chw(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # Inverse of _to_hwc using the reference shape to restore C or (C,K)
    if ref.ndim == 3:  # (C,H,W)
        return np.moveaxis(x, -1, 0)
    elif ref.ndim == 4:  # (C,H,W,K)
        c, h, w, k = ref.shape
        x = x.reshape(h, w, c, k).transpose(2, 0, 1, 3)
        return x
    else:
        raise ValueError(f"Unexpected ref.ndim {ref.ndim}")
