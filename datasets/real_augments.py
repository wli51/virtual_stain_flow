"""
/datasets/real_augments.py

This file implements real augmentations for bounding boxes, currently 
rotation and translation within the original image FOV.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict

import pandas as pd
import numpy as np
from numpy.random import Generator
from scipy.special import softmax

from .bbox_schema import BBoxSchema

"""
helpers
"""
def roundcast(arr: np.ndarray, mode: str = "round") -> np.ndarray:
    modes = {"round": np.rint, "floor": np.floor, "ceil": np.ceil}
    if mode not in modes:
        raise ValueError(f"rounding must be one of {list(modes.keys())}")
    return modes[mode](arr).astype(np.int64)

def ensure_array_broadcast(value: Union[int, np.ndarray], length: int, dtype=int) -> np.ndarray:
    if isinstance(value, int):
        return np.full(length, value, dtype=dtype)
    return np.asarray(value, dtype=dtype)

def compute_weighted_softmax(limits_df: pd.DataFrame, weight_cols: List[str]) -> np.ndarray:
    if len(weight_cols) == 1:
        weights = np.abs(limits_df[weight_cols[0]].to_numpy(dtype=float))
    else:
        weights = sum(np.abs(limits_df[col].to_numpy(dtype=float)) for col in weight_cols)
    return softmax(weights)

"""
Real Augmentations
"""
class BaseAugmentation:
    name: str
    def compute_limits(self, df: pd.DataFrame, schema: BBoxSchema, **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        raise NotImplementedError
    def sample_params(self, gen: Generator, limits_sel: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError
    def materialize(self, schema: BBoxSchema, source_rows: pd.DataFrame, params: Dict[str, np.ndarray], rounding: str = "round") -> pd.DataFrame:
        raise NotImplementedError
    def _sample_between(
        self,
        n: int,
        lo: np.ndarray,
        hi: np.ndarray,
        *,
        gen: Optional[Generator] = None,
        keyed_info: Optional[Dict[str, np.ndarray | int]] = None,
        additional_key: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Helper for drawing n samples in [lo, hi) per row.
        - If keyed_info provided -> per-row keyed RNG.
        - Else -> use provided Generator gen (required).
        """
        span = np.maximum(hi - lo, 0.0)        
        if keyed_info is None:
            u = gen.random(n)
        else: 
            u = np.empty(n, dtype=float)
            base_seed = int(keyed_info["base_seed"])
            aug_index = int(keyed_info["aug_index"])
            parents = np.asarray(keyed_info["parent_indices"], dtype=np.int64)
            occ = np.asarray(keyed_info["occurrence_ranks"], dtype=np.int64)
            for i in range(n):
                rng = keyed_rng(
                    base_seed, 
                    # this ensures variation for multiple parameter sampled in same augmetnation
                    aug_index * 2 + additional_key, 
                    int(parents[i]), 
                    int(occ[i])
                )
                u[i] = rng.random()
        return lo + u * span

    def _add_augmentation_metadata(self, df: pd.DataFrame, params: Dict[str, np.ndarray]) -> pd.DataFrame:
        df["_aug_type"] = self.name
        for key, value in params.items():
            df[key] = value
        return df

@dataclass(frozen=True)
class RotationAug(BaseAugmentation):
    name: str = "rotation"
    step_deg: float = 1.0
    max_deg: float = 90.0
    eps: float = 1e-9
    default_chunk_size: int = 1024

    def _compute_rotation_limits(
        self,
        xmin: np.ndarray, ymin: np.ndarray, xmax: np.ndarray, ymax: np.ndarray,
        cx: np.ndarray, cy: np.ndarray, fov_h: np.ndarray, fov_w: np.ndarray,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Vectorized computation of rotation limits for bounding boxes.
        """
        # this is kind of sloppy
        xmin = np.asarray(xmin, dtype=np.float64).ravel()
        ymin = np.asarray(ymin, dtype=np.float64).ravel()
        xmax = np.asarray(xmax, dtype=np.float64).ravel()
        ymax = np.asarray(ymax, dtype=np.float64).ravel()
        cx   = np.asarray(cx, dtype=np.float64).ravel()
        cy   = np.asarray(cy, dtype=np.float64).ravel()
        fov_h = np.asarray(fov_h, dtype=np.float64).ravel()
        fov_w = np.asarray(fov_w, dtype=np.float64).ravel()
    
        N = xmin.shape[0]
        if not all(a.shape == (N,) for a in (
            ymin, xmax, ymax, cx, cy, fov_h, fov_w)):
            raise ValueError("All input arrays must have the same length.")
        
        # A rotated box safely contained in a FOV if all its corners are 
        # inside the FOV ,so we compute and rotated corners for each bbox to 
        # determine the limits. Here we first represent boxes by their corners
        corners = np.stack([
            np.stack([xmin, ymin], axis=1),
            np.stack([xmax, ymin], axis=1),
            np.stack([xmax, ymax], axis=1),
            np.stack([xmin, ymax], axis=1),
        ], axis=1)  # (N,4,2)

        # and center of rotations provided
        centers = np.stack([cx, cy], axis=1)[:, None, :]  # (N,2) -> (N,1,2)

        # discrete angles of rotation to check safe rotation range
        # per direction of rotation, so with both clockwise and counter-clockwise
        # rotations we check 2 * n_steps angles
        n_steps = int(np.floor(self.max_deg / self.step_deg + self.eps))
        ang_pos = np.arange(1, n_steps + 1, dtype=np.float64) * self.step_deg
        ang_neg = -ang_pos # (M, )

        def rot_mats(angles):
            th = np.deg2rad(angles)
            cos_t, sin_t = np.cos(th), np.sin(th)
            R = np.stack([
                np.stack([cos_t, -sin_t], axis=-1), 
                np.stack([sin_t,  cos_t], axis=-1)
            ], axis=-2)
            return R
        R_pos = rot_mats(ang_pos)  # (M, 2, 2)
        R_neg = rot_mats(ang_neg)  # (M, 2, 2)

        # this method does batched vectorization to avoid using too much memory
        def batched_compute_limit(
            start: int,
            end: int,
            r_mats: np.ndarray, # must be in the order of increasing mangnitude
        ):
            _corners = corners[start:end] # (B, 4, 2)
            _centers = centers[start:end] # (B, 1, 2)
            _fov_h = fov_h[start:end, None, None] # (B, ) -> (B, 1, 1)
            _fov_w = fov_w[start:end, None, None] # (B, ) -> (B, 1, 1)

            corners_rotated = np.einsum( # centered rotation of corners
                'bfd,mde->bmfe', _corners - _centers, r_mats 
            ) + _centers[:, None, :]  # (B, M, 4, 2)
            x, y = corners_rotated[..., 0], corners_rotated[..., 1]  # (B, M, 4)

            # check box containment in FOV per all boxes (B) and all degrees (M)
            corners_inside_fov = (x >= -self.eps) & (x <= _fov_w - self.eps) & \
                (y >= -self.eps) & (y <= _fov_h - self.eps)  # (B, M, 4)
            box_inside_fov = np.all(corners_inside_fov, axis=-1)  # (B, M)

            # compute largest degree of rotation allowed (for direction)
            max_valid_steps = np.cumprod(box_inside_fov, axis=1).sum(axis=1)  # (B,)
            return max_valid_steps.astype(np.float64) * self.step_deg
        
        if chunk_size is None:
            chunk_size = N
        max_pos_all = np.empty((N,), dtype=np.float64)
        max_neg_all = np.empty((N,), dtype=np.float64)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            max_pos_all[s:e] = batched_compute_limit(s, e, R_pos)
            max_neg_all[s:e] = batched_compute_limit(s, e, R_neg)
        
        return pd.DataFrame({
            'min_angle': -max_neg_all,
            'max_angle': max_pos_all
        })

    def compute_limits(
        self,
        df: pd.DataFrame,
        schema: BBoxSchema,
        fov_h: Union[int, np.ndarray] = 1080,
        fov_w: Union[int, np.ndarray] = 1080,
    ) -> Tuple[pd.DataFrame, np.ndarray]:        
        limits = self._compute_rotation_limits(
            xmin=df[schema.x_min].to_numpy(float),
            ymin=df[schema.y_min].to_numpy(float),
            xmax=df[schema.x_max].to_numpy(float),
            ymax=df[schema.y_max].to_numpy(float),
            cx=df[schema.cx].to_numpy(float),
            cy=df[schema.cy].to_numpy(float),
            fov_h=ensure_array_broadcast(fov_h, len(df), dtype=float),
            fov_w=ensure_array_broadcast(fov_w, len(df), dtype=float),
            chunk_size=self.default_chunk_size
        )
        weights = compute_weighted_softmax(limits, ['max_angle', 'min_angle'])
        return limits, weights

    def sample_params(
        self,
        limits_sel: pd.DataFrame,
        *,
        gen: Optional[Generator] = None,
        keyed_info: Optional[Dict[str, np.ndarray | int]] = None
    ) -> Dict[str, np.ndarray]:
        lo = limits_sel["min_angle"].to_numpy(float)
        hi = limits_sel["max_angle"].to_numpy(float)
        n = len(limits_sel)
        return {'_sampled_angle': self._sample_between(
            n=n, lo=lo, hi=hi,gen=gen, keyed_info=keyed_info
        )}

    def materialize(
        self,
        schema: BBoxSchema,
        source_rows: pd.DataFrame,
        params: Dict[str, np.ndarray],
        rounding: str = "round"
    ) -> pd.DataFrame:
        augs = source_rows.copy()
        if not np.issubdtype(augs[schema.angle].dtype, np.floating):
            augs[schema.angle] = augs[schema.angle].astype(float)
        augs[schema.angle] += params["_sampled_angle"]
        return self._add_augmentation_metadata(augs, params)

@dataclass(frozen=True)
class TranslationAug(BaseAugmentation):
    name: str = "translation"

    def _compute_translation_limits(
        self,
        bbox_coords: Dict[str, np.ndarray],
        fov_shape: Tuple[np.ndarray, np.ndarray],
        max_translation: Dict[str, Optional[float]]
    ) -> pd.DataFrame:
        xmin, ymin, xmax, ymax = bbox_coords.values()
        fov_h, fov_w = fov_shape
        w = (xmax - xmin).astype(float)
        h = (ymax - ymin).astype(float)

        max_tx = (0.45 * w if max_translation['x'] is None
                  else ensure_array_broadcast(max_translation['x'], len(w), dtype=float))
        max_ty = (0.45 * h if max_translation['y'] is None
                  else ensure_array_broadcast(max_translation['y'], len(h), dtype=float))

        return pd.DataFrame({
            "dx_min": np.maximum(-xmin, -max_tx),
            "dx_max": np.minimum(fov_w - xmax, max_tx),
            "dy_min": np.maximum(-ymin, -max_ty),
            "dy_max": np.minimum(fov_h - ymax, max_ty)
        })

    def compute_limits(
        self,
        df: pd.DataFrame,
        schema: BBoxSchema,
        fov_h: Union[int, np.ndarray] = 1080,
        fov_w: Union[int, np.ndarray] = 1080,
        max_translation_x: Optional[float] = None,
        max_translation_y: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        fov_h = ensure_array_broadcast(fov_h, len(df))
        fov_w = ensure_array_broadcast(fov_w, len(df))
        df = df.copy().bbox(schema).ensure_columns()

        bbox_coords = {
            'xmin': df[schema.x_min].to_numpy(),
            'ymin': df[schema.y_min].to_numpy(),
            'xmax': df[schema.x_max].to_numpy(),
            'ymax': df[schema.y_max].to_numpy()
        }

        limits = self._compute_translation_limits(
            bbox_coords,
            (fov_h, fov_w),
            {'x': max_translation_x, 'y': max_translation_y}
        )
        weights = compute_weighted_softmax(limits, ['dx_max', 'dx_min', 'dy_max', 'dy_min'])
        return limits, weights

    def sample_params(
        self,
        limits_sel: pd.DataFrame,
        *,
        gen: Optional[Generator] = None,
        keyed_info: Optional[Dict[str, np.ndarray | int]] = None
    ) -> Dict[str, np.ndarray]:

        sampled = {}
        n = len(limits_sel)
        lo_dx, hi_dx = limits_sel["dx_min"].to_numpy(float), limits_sel["dx_max"].to_numpy(float)
        lo_dy, hi_dy = limits_sel["dy_min"].to_numpy(float), limits_sel["dy_max"].to_numpy(float)

        sampled['_sampled_dx'] = self._sample_between(
            n=n, lo=lo_dx, hi=hi_dx, 
            gen=gen, keyed_info=keyed_info, additional_key=0)
        sampled['_sampled_dy'] = self._sample_between(
            n=n, lo=lo_dy, hi=hi_dy, 
            gen=gen, keyed_info=keyed_info, additional_key=1)

        return sampled
    
    def materialize(
        self, schema: BBoxSchema, source_rows: pd.DataFrame,
        params: Dict[str, np.ndarray], rounding: str = "round"
    ) -> pd.DataFrame:
        augs = source_rows.copy()

        coords = {col: augs[getattr(schema, col)].to_numpy(dtype=np.int64)
                  for col in ['x_min', 'x_max', 'y_min', 'y_max']}
        dx, dy = params["_sampled_dx"], params["_sampled_dy"]

        augs[schema.x_min] = roundcast(coords['x_min'] + dx, rounding)
        augs[schema.x_max] = roundcast(coords['x_max'] + dx, rounding)
        augs[schema.y_min] = roundcast(coords['y_min'] + dy, rounding)
        augs[schema.y_max] = roundcast(coords['y_max'] + dy, rounding)

        if schema.rot_cx in augs.columns and schema.rot_cy in augs.columns:
            augs[schema.rot_cx] = augs[schema.rot_cx].to_numpy(dtype=float) + dx
            augs[schema.rot_cy] = augs[schema.rot_cy].to_numpy(dtype=float) + dy

        return self._add_augmentation_metadata(augs, params)