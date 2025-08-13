from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence, Union, Dict
import pathlib
import hashlib

import pandas as pd
import numpy as np
from numpy.random import PCG64, Generator, SeedSequence
from scipy.special import softmax

from .bbox_schema import BBoxSchema

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

def spawn_generators(n: int, seed: int) -> List[Generator]:
    ss = SeedSequence(seed)
    return [Generator(PCG64(s)) for s in ss.spawn(n)]

def diagnose_generator_independence(generators: List[Generator], n_samples: int = 1000) -> Dict[str, float]:
    if len(generators) < 2:
        return {}
    samples_0 = generators[0].random(n_samples)
    samples_1 = generators[1].random(n_samples)
    correlation = np.corrcoef(samples_0, samples_1)[0, 1]
    return {
        'correlation_gen0_gen1': correlation,
        'mean_gen0': np.mean(samples_0),
        'mean_gen1': np.mean(samples_1),
        'std_gen0': np.std(samples_0),
        'std_gen1': np.std(samples_1)
    }

def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p[~np.isfinite(p)] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))

def weighted_choice(gen: Generator, p: np.ndarray, size: int, replace: bool = True) -> np.ndarray:
    p = normalize_probs(p)
    n = len(p)
    if not replace and size > n:
        replace = True
    return gen.choice(n, size=size, replace=replace, p=p)

def normalize_pathlike(x: pathlib.Path, case_sensitive: bool = True) -> str:
    return x.as_posix().lower() if not case_sensitive else x.as_posix()

def keyed_rng(*keys: int) -> Generator:
    """
    Construct a deterministic RNG keyed by a tuple of integers.
    Output is independent of iteration order and only depends on the keys.
    """
    h = hashlib.blake2b(digest_size=16)
    for k in keys:
        h.update(int(k).to_bytes(8, "little", signed=False))
    seed_int = int.from_bytes(h.digest(), "little")
    ss = SeedSequence(seed_int)
    return Generator(PCG64(ss))

# ----------------------------
# Aug base + concrete augs
# ----------------------------

class BaseAugmentation:
    name: str
    def compute_limits(self, df: pd.DataFrame, schema: BBoxSchema, **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        raise NotImplementedError
    def sample_params(self, gen: Generator, limits_sel: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError
    def materialize(self, schema: BBoxSchema, source_rows: pd.DataFrame, params: Dict[str, np.ndarray], rounding: str = "round") -> pd.DataFrame:
        raise NotImplementedError
    def _add_augmentation_metadata(self, df: pd.DataFrame, params: Dict[str, np.ndarray]) -> pd.DataFrame:
        df["_aug_type"] = self.name
        for key, value in params.items():
            df[key] = value
        return df

# ---- Rotation ----
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
        gen: Generator,                         # kept for API symmetry; unused by keyed mode
        limits_sel: pd.DataFrame,
        *,
        keyed_info: Optional[Dict[str, np.ndarray | int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        If keyed_info is provided, use keyed RNG per (seed, aug_index, parent, occurrence).
        keyed_info must contain:
           - base_seed: int
           - aug_index: int
           - parent_indices: np.ndarray[int]  (src_idx for these rows)
           - occurrence_ranks: np.ndarray[int] (0,1,2,...) per parent
        """
        lo = limits_sel["min_angle"].to_numpy(float)
        hi = limits_sel["max_angle"].to_numpy(float)
        n = len(limits_sel)

        if keyed_info is None:
            # fallback: plain RNG
            u = gen.random(n)
        else:
            base_seed = int(keyed_info["base_seed"])       # type: ignore
            aug_index = int(keyed_info["aug_index"])       # type: ignore
            parents = np.asarray(keyed_info["parent_indices"], dtype=np.int64)     # type: ignore
            occ = np.asarray(keyed_info["occurrence_ranks"], dtype=np.int64)       # type: ignore
            # one uniform per row using keyed RNG
            u = np.empty(n, dtype=float)
            for i in range(n):
                rng = keyed_rng(base_seed, aug_index, int(parents[i]), int(occ[i]))
                u[i] = rng.random()

        angle = lo + u * np.maximum(hi - lo, 0.0)
        return {"_sampled_angle": angle}

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

# ---- Translation ----
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
        gen: Generator,                         # kept for API symmetry; unused by keyed mode
        limits_sel: pd.DataFrame,
        *,
        keyed_info: Optional[Dict[str, np.ndarray | int]] = None
    ) -> Dict[str, np.ndarray]:

        n = len(limits_sel)
        lo_dx, hi_dx = limits_sel["dx_min"].to_numpy(float), limits_sel["dx_max"].to_numpy(float)
        lo_dy, hi_dy = limits_sel["dy_min"].to_numpy(float), limits_sel["dy_max"].to_numpy(float)

        if keyed_info is None:
            u_x = gen.random(n)
            u_y = gen.random(n)
        else:
            base_seed = int(keyed_info["base_seed"])       # type: ignore
            aug_index = int(keyed_info["aug_index"])       # type: ignore
            parents = np.asarray(keyed_info["parent_indices"], dtype=np.int64)     # type: ignore
            occ = np.asarray(keyed_info["occurrence_ranks"], dtype=np.int64)       # type: ignore

            # Different salt for x/y by perturbing aug_index (keeps API compact)
            u_x = np.empty(n, dtype=float)
            u_y = np.empty(n, dtype=float)
            for i in range(n):
                rngx = keyed_rng(base_seed, aug_index * 2 + 0, int(parents[i]), int(occ[i]))
                rngy = keyed_rng(base_seed, aug_index * 2 + 1, int(parents[i]), int(occ[i]))
                u_x[i] = rngx.random()
                u_y[i] = rngy.random()

        dx = lo_dx + u_x * np.maximum(hi_dx - lo_dx, 0.0)
        dy = lo_dy + u_y * np.maximum(hi_dy - lo_dy, 0.0)
        return {"_sampled_dx": dx, "_sampled_dy": dy}

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

# ----------------------------
# Deterministic sorting (unchanged)
# ----------------------------

def deterministic_sort_metadata(
    file_index: pd.DataFrame,
    metadata: pd.DataFrame,
    id_cols: Sequence[str],
    case_sensitive: bool = True,
    na_position: str = "last"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if len(file_index) != len(metadata):
        raise ValueError("file_index and metadata must have the same length")

    file_index_norm = file_index.copy().map(
        lambda p: normalize_pathlike(p, case_sensitive=case_sensitive)
        if isinstance(p, pathlib.Path) else p
    )
    key_df = pd.concat([file_index_norm, metadata.loc[:, id_cols]], axis=1)

    order = key_df.sort_values(list(key_df.columns),
                               kind="mergesort",
                               na_position=na_position).index

    file_index_sorted = file_index.loc[order].reset_index(drop=True)
    metadata_sorted = metadata.loc[order].reset_index(drop=True)
    return file_index_sorted, metadata_sorted

# ----------------------------
# Planner & pipeline
# ----------------------------

class AugPlanConfig:
    def __init__(
        self,
        seed: int = 42,
        rounding: str = "round",
        replace_per_aug: bool = True,
        weights_per_aug: Optional[List[float]] = None
    ):
        self.seed = seed
        self.rounding = rounding
        self.replace_per_aug = replace_per_aug
        if weights_per_aug is None:
            weights_per_aug = []
        self.weights_per_aug = np.asarray(weights_per_aug, dtype=float)

def allocate_augmentation_slots(
    n_slots: int,
    aug_names: List[str],
    weights: Optional[List[float]] = None,
    seed: int = 42
) -> np.ndarray:
    K = len(aug_names)
    if weights is None:
        weights = np.ones(K, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != K:
        raise ValueError("weights and names must be the same length")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("Sum of weights must be > 0")

    probs = weights / total
    gen = spawn_generators(1, seed)[0]
    u = gen.random(n_slots)
    cum = np.cumsum(probs)
    slot_types = np.searchsorted(cum, u, side="right")
    return np.clip(slot_types, 0, K - 1)

def make_parent_key(file_index_df: pd.DataFrame, metadata_df: pd.DataFrame, key_cols: Sequence[str]) -> pd.Series:
    finfo = file_index_df.astype(str)
    key_df = pd.concat([finfo, metadata_df.loc[:, key_cols]], axis=1)
    return key_df.astype(str).agg("||".join, axis=1)

def _occurrence_ranks(indices: np.ndarray) -> np.ndarray:
    """
    For a sequence of parent indices (src_idx), return occurrence rank per position:
    e.g., [5, 2, 5, 5, 2] -> [0, 0, 1, 2, 1]
    """
    seen: Dict[int, int] = {}
    ranks = np.empty_like(indices, dtype=int)
    for i, p in enumerate(indices):
        c = seen.get(int(p), 0)
        ranks[i] = c
        seen[int(p)] = c + 1
    return ranks

def run_augmentations(
    file_index_sorted: pd.DataFrame,
    metadata_sorted: pd.DataFrame,
    schema: BBoxSchema,
    augmentations: List[BaseAugmentation],
    n_slots: int,
    config: AugPlanConfig,
    key_cols_for_parent: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    N = len(metadata_sorted)
    assert len(file_index_sorted) == N

    _ = metadata_sorted.bbox(schema).ensure_columns()

    # 1) per-aug limits + weights
    limits_list: List[pd.DataFrame] = []
    prob_list: List[np.ndarray] = []
    for aug in augmentations:
        limits, p = aug.compute_limits(metadata_sorted, schema)
        limits_list.append(limits.reset_index(drop=True))
        prob_list.append(p)

    # 2) allocate slots across aug types
    aug_names = [a.name for a in augmentations]
    slot_types = allocate_augmentation_slots(
        n_slots, aug_names, config.weights_per_aug, seed=config.seed)
    per_type_indices = [np.where(slot_types == k)[0] for k in range(len(augmentations))]

    # Independent RNGs for index selection (parameter RNG is KEYED, not stream-based)
    all_generators = spawn_generators(len(augmentations), config.seed)
    g_idx = all_generators  # one generator per aug type for choosing sources

    aug_rows = []
    lineage_cols_common = [
        "_source_kind","_aug_type","_aug_slot","_parent_src_index","_parent_key",
        "_sampled_angle","_sampled_dx","_sampled_dy"
    ]

    parent_key_series = make_parent_key(file_index_sorted, metadata_sorted, key_cols_for_parent)
    parent_key_arr = parent_key_series.to_numpy()

    for k, aug in enumerate(augmentations):
        slot_ids = per_type_indices[k]
        if len(slot_ids) == 0:
            continue
        n_k = len(slot_ids)
        p = prob_list[k]
        limits = limits_list[k]

        # choose sources (reproducible via g_idx[k])
        src_idx = weighted_choice(g_idx[k], p, size=n_k, replace=config.replace_per_aug)

        # grab limits for these sources
        sel_limits = limits.iloc[src_idx].reset_index(drop=True)

        # compute occurrence ranks per parent for keyed RNG (0,1,2,...)
        occ_ranks = _occurrence_ranks(src_idx)

        # ---- KEYED PARAMETER SAMPLING (Approach 2) ----
        keyed_info = {
            "base_seed": int(config.seed),
            "aug_index": int(k),
            "parent_indices": src_idx.astype(np.int64),
            "occurrence_ranks": occ_ranks.astype(np.int64),
        }
        params = aug.sample_params(None, sel_limits, keyed_info=keyed_info)  # gen is unused in keyed mode

        # materialize
        src_rows = metadata_sorted.iloc[src_idx].reset_index(drop=True)
        augs_k = aug.materialize(schema, src_rows, params, rounding=config.rounding)

        # lineage
        augs_k["_source_kind"] = "augmentation"
        augs_k["_aug_type"] = aug.name
        augs_k["_aug_slot"] = np.nan
        augs_k["_parent_src_index"] = src_idx
        augs_k["_parent_key"] = parent_key_arr[src_idx]
        for col in ("_sampled_angle","_sampled_dx","_sampled_dy"):
            if col not in augs_k.columns:
                augs_k[col] = np.nan

        aug_rows.append(augs_k)

    aug_df = pd.concat(aug_rows, axis=0, ignore_index=True) if aug_rows else pd.DataFrame(
        columns=metadata_sorted.columns.tolist() + lineage_cols_common
    )

    if not aug_df.empty:
        aug_df["_tmp_row"] = np.arange(len(aug_df))
        aug_df["_aug_slot"] = (
            aug_df["_tmp_row"].groupby(aug_df["_parent_src_index"]).rank(method="first").astype(int) - 1
        )
        aug_df = aug_df.drop(columns=["_tmp_row"])

    originals = metadata_sorted.copy()
    originals["_source_kind"] = "original"
    originals["_aug_type"] = np.nan
    originals["_aug_slot"] = np.nan
    originals["_parent_src_index"] = np.arange(N, dtype=int)
    originals["_parent_key"] = parent_key_arr
    for col in ("_sampled_angle","_sampled_dx","_sampled_dy"):
        originals[col] = np.nan

    if not aug_df.empty:
        file_index_augs = file_index_sorted.iloc[aug_df["_parent_src_index"].to_numpy()].reset_index(drop=True)
        file_index_augs.columns = file_index_sorted.columns
        file_index_merged = pd.concat([file_index_sorted.reset_index(drop=True), file_index_augs],
                                      axis=0, ignore_index=True)
        metadata_merged = pd.concat([originals.reset_index(drop=True), aug_df.reset_index(drop=True)],
                                    axis=0, ignore_index=True)

        helper = metadata_merged.assign(
            __kind_rank = metadata_merged["_source_kind"].map({"original":0,"augmentation":1}).astype(int),
            __slot_rank = metadata_merged["_aug_slot"].fillna(-1).astype(int)
        )
        order = helper.sort_values(["_parent_key","__kind_rank","__slot_rank"], kind="mergesort").index
        file_index_merged = file_index_merged.loc[order].reset_index(drop=True)
        metadata_merged   = metadata_merged.loc[order].reset_index(drop=True)
    else:
        file_index_merged = file_index_sorted.reset_index(drop=True)
        metadata_merged   = originals.reset_index(drop=True)

    parent_pos = (
        metadata_merged.reset_index().groupby("_parent_key")["index"].min().rename("_parent_merged_index")
    )
    metadata_merged = metadata_merged.merge(parent_pos, left_on="_parent_key", right_index=True, how="left")

    return aug_df, file_index_merged, metadata_merged