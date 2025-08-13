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
    """Cast array to int64 with specified rounding mode."""
    modes = {
        "round": np.rint,
        "floor": np.floor, 
        "ceil": np.ceil
    }
    if mode not in modes:
        raise ValueError(f"rounding must be one of {list(modes.keys())}")
    return modes[mode](arr).astype(np.int64)

def ensure_array_broadcast(value: Union[int, np.ndarray], length: int, dtype=int) -> np.ndarray:
    """Broadcast scalar or validate array to match expected length."""
    if isinstance(value, int):
        return np.full(length, value, dtype=dtype)
    return np.asarray(value, dtype=dtype)

def compute_weighted_softmax(limits_df: pd.DataFrame, weight_cols: List[str]) -> np.ndarray:
    """Compute softmax weights from limits DataFrame columns."""
    if len(weight_cols) == 1:
        weights = np.abs(limits_df[weight_cols[0]].to_numpy(dtype=float))
    else:
        weights = sum(np.abs(limits_df[col].to_numpy(dtype=float)) for col in weight_cols)
    return softmax(weights)

# ---- Base interface ----
class BaseAugmentation:
    """
    Base class for geometric augmentations with consistent interface:
    - compute_limits: Calculate valid parameter ranges per sample
    - sample_params: Generate random parameters within limits  
    - materialize: Apply augmentation to bounding boxes
    """
    name: str

    def compute_limits(self, df: pd.DataFrame, schema: BBoxSchema, **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        raise NotImplementedError

    def sample_params(self, gen: Generator, limits_sel: pd.DataFrame) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def materialize(
        self,
        schema: BBoxSchema,
        source_rows: pd.DataFrame,
        params: Dict[str, np.ndarray],
        rounding: str = "round",
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _add_augmentation_metadata(self, df: pd.DataFrame, params: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Add common augmentation metadata to output DataFrame."""
        df["_aug_type"] = self.name
        for key, value in params.items():
            df[key] = value
        return df

# ---- Concrete: Rotation ----
@dataclass(frozen=True)
class RotationAug(BaseAugmentation):
    name: str = "rotation"
    step_deg: float = 1.0

    def compute_limits(
        self, 
        df: pd.DataFrame, 
        schema: BBoxSchema,
        fov_h: Union[int, np.ndarray] = 1080,
        fov_w: Union[int, np.ndarray] = 1080,
    ) -> Tuple[pd.DataFrame, np.ndarray]:        

        if isinstance(fov_h, int):
            fov_h = np.full(len(df), fov_h, dtype=int)
        if isinstance(fov_w, int):
            fov_w = np.full(len(df), fov_w, dtype=int)
        
def compute_rotation_matrix(angle_deg: float) -> np.ndarray:
    """Compute 2D rotation matrix for given angle in degrees."""
    theta = np.deg2rad(angle_deg)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ], dtype=np.float64)

def rotate_corners(corners: np.ndarray, angle_deg: float, center: np.ndarray) -> np.ndarray:
    """Rotate corner points around a center point."""
    rot_matrix = compute_rotation_matrix(angle_deg)
    shifted = corners - center
    return shifted @ rot_matrix.T + center

def corners_inside_fov(corners: np.ndarray, fov_w: int, fov_h: int) -> bool:
    """Check if all corners are inside field of view."""
    return np.all((0 <= corners[:, 0]) & (corners[:, 0] < fov_w) &
                  (0 <= corners[:, 1]) & (corners[:, 1] < fov_h))

def find_max_rotation_range(bbox: Tuple[int, int, int, int], center: Tuple[float, float], 
                           fov_shape: Tuple[int, int], step_deg: float = 1.0) -> Tuple[int, int]:
    """Find maximum safe rotation range for a bounding box."""
    xmin, ymin, xmax, ymax = bbox
    fov_h, fov_w = fov_shape
    
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    center_point = np.array([[center[0], center[1]]], dtype=np.float64)
    
    n_steps = int(np.floor(180.0 / step_deg + 1e-12))
    
    # Find maximum positive rotation
    k_pos = 0
    for k in range(1, n_steps + 1):
        angle = k * step_deg
        if corners_inside_fov(rotate_corners(corners, angle, center_point), fov_w, fov_h):
            k_pos = k
        else:
            break
    
    # Find maximum negative rotation  
    k_neg = 0
    for k in range(1, n_steps + 1):
        angle = -k * step_deg
        if corners_inside_fov(rotate_corners(corners, angle, center_point), fov_w, fov_h):
            k_neg = k
        else:
            break
    
    return int(np.floor(-k_neg * step_deg)), int(np.ceil(k_pos * step_deg))
# ---- Concrete: Rotation ----
@dataclass(frozen=True)
class RotationAug(BaseAugmentation):
    name: str = "rotation"
    step_deg: float = 1.0

    def compute_limits(
        self, 
        df: pd.DataFrame, 
        schema: BBoxSchema,
        fov_h: Union[int, np.ndarray] = 1080,
        fov_w: Union[int, np.ndarray] = 1080,
    ) -> Tuple[pd.DataFrame, np.ndarray]:        

        fov_h = ensure_array_broadcast(fov_h, len(df))
        fov_w = ensure_array_broadcast(fov_w, len(df))
        
        df = df.copy().bbox(schema).ensure_columns()
        bbox_acc = df.bbox(schema)

        limits_data = []
        for i in range(len(df)):
            row = bbox_acc.row(i)
            min_angle, max_angle = find_max_rotation_range(
                bbox=row.bbox,
                center=row.rot_center, 
                fov_shape=(fov_h[i], fov_w[i]),
                step_deg=self.step_deg
            )            
            limits_data.append([min_angle, max_angle])
        
        limits = pd.DataFrame(limits_data, columns=['min_angle', 'max_angle'])
        weights = compute_weighted_softmax(limits, ['max_angle', 'min_angle'])
        return limits, weights

    def sample_params(self, gen: Generator, limits_sel: pd.DataFrame) -> Dict[str, np.ndarray]:
        lo = limits_sel["min_angle"].to_numpy(dtype=float)
        hi = limits_sel["max_angle"].to_numpy(dtype=float)
        u = gen.random(len(limits_sel))
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
        
        # Ensure angle column is float and add sampled angle
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
        """Compute safe translation limits for bounding boxes."""
        xmin, ymin, xmax, ymax = bbox_coords.values()
        fov_h, fov_w = fov_shape
        
        w = (xmax - xmin).astype(float)
        h = (ymax - ymin).astype(float)
        
        # Resolve per-axis max translations
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

    def sample_params(self, gen: Generator, limits_sel: pd.DataFrame) -> Dict[str, np.ndarray]:
        n_samples = len(limits_sel)
        
        # Use sequential sampling from the same generator to maintain determinism
        # but reduce correlation between x and y by using different random draws
        u_x = gen.random(n_samples)  # First n_samples draws for x
        u_y = gen.random(n_samples)  # Next n_samples draws for y
        
        # Apply to limits
        lo_dx, hi_dx = limits_sel["dx_min"].to_numpy(float), limits_sel["dx_max"].to_numpy(float)
        lo_dy, hi_dy = limits_sel["dy_min"].to_numpy(float), limits_sel["dy_max"].to_numpy(float)
        
        dx = lo_dx + u_x * np.maximum(hi_dx - lo_dx, 0.0)
        dy = lo_dy + u_y * np.maximum(hi_dy - lo_dy, 0.0)
        
        return {"_sampled_dx": dx, "_sampled_dy": dy}
    
    def _sample_uniform_range(self, gen: Generator, limits: pd.DataFrame, 
                            lo_col: str, hi_col: str, n_samples: int) -> np.ndarray:
        """Sample uniformly between lo and hi columns."""
        lo = limits[lo_col].to_numpy(float)
        hi = limits[hi_col].to_numpy(float)
        u = gen.random(n_samples)
        return lo + u * np.maximum(hi - lo, 0.0)

    def materialize(
        self, schema: BBoxSchema, source_rows: pd.DataFrame, 
        params: Dict[str, np.ndarray], rounding: str = "round"
    ) -> pd.DataFrame:
        augs = source_rows.copy()
        
        # Extract coordinates and translations
        coords = {col: augs[getattr(schema, col)].to_numpy(dtype=np.int64) 
                 for col in ['x_min', 'x_max', 'y_min', 'y_max']}
        dx, dy = params["_sampled_dx"], params["_sampled_dy"]
        
        # Apply translations with rounding
        augs[schema.x_min] = roundcast(coords['x_min'] + dx, rounding)
        augs[schema.x_max] = roundcast(coords['x_max'] + dx, rounding) 
        augs[schema.y_min] = roundcast(coords['y_min'] + dy, rounding)
        augs[schema.y_max] = roundcast(coords['y_max'] + dy, rounding)
        
        # Update rotation centers if present
        if schema.rot_cx in augs.columns and schema.rot_cy in augs.columns:
            augs[schema.rot_cx] = augs[schema.rot_cx].to_numpy(dtype=float) + dx
            augs[schema.rot_cy] = augs[schema.rot_cy].to_numpy(dtype=float) + dy
        
        return self._add_augmentation_metadata(augs, params)   

"""
Reproducible planner
"""
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

def spawn_generators(n: int, seed: int) -> List[Generator]:
    """Create n independent random number generators from a seed."""
    ss = SeedSequence(seed)
    return [Generator(PCG64(s)) for s in ss.spawn(n)]

def diagnose_generator_independence(generators: List[Generator], n_samples: int = 1000) -> Dict[str, float]:
    """
    Diagnostic function to check independence between generators.
    Returns correlation coefficients between first few generators.
    """
    if len(generators) < 2:
        return {}
    
    # Generate samples from first two generators
    samples_0 = generators[0].random(n_samples)
    samples_1 = generators[1].random(n_samples)
    
    # Compute correlation
    correlation = np.corrcoef(samples_0, samples_1)[0, 1]
    
    return {
        'correlation_gen0_gen1': correlation,
        'mean_gen0': np.mean(samples_0),
        'mean_gen1': np.mean(samples_1),
        'std_gen0': np.std(samples_0), 
        'std_gen1': np.std(samples_1)
    }

def normalize_probs(p: np.ndarray) -> np.ndarray:
    """Normalize probability array, handling edge cases."""
    p = np.asarray(p, dtype=float)
    p[~np.isfinite(p)] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))

def weighted_choice(gen: Generator, p: np.ndarray, size: int, replace: bool = True) -> np.ndarray:
    """Sample indices with given probabilities."""
    p = normalize_probs(p)
    n = len(p)
    if not replace and size > n:
        replace = True  # fallback if impossible
    return gen.choice(n, size=size, replace=replace, p=p)

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

def allocate_augmentation_slots(
    n_slots: int, 
    aug_names: List[str], 
    weights: Optional[List[float]] = None,
    seed: int = 42
) -> np.ndarray:
    """Allocate slots to augmentation types using deterministic sampling."""
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
    
    # Vectorized inverse-CDF
    slot_types = np.searchsorted(cum, u, side="right")
    return np.clip(slot_types, 0, K-1)

def make_parent_key(file_index_df: pd.DataFrame, metadata_df: pd.DataFrame, key_cols: Sequence[str]) -> pd.Series:
    # Normalize paths to string; concat with key cols (stable)
    finfo = file_index_df.astype(str)
    key_df = pd.concat([finfo, metadata_df.loc[:, key_cols]], axis=1)
    return key_df.astype(str).agg("||".join, axis=1)

def run_augmentations(
    file_index_sorted: pd.DataFrame,
    metadata_sorted: pd.DataFrame,
    schema: BBoxSchema,
    augmentations: List[BaseAugmentation],
    n_slots: int,
    config: AugPlanConfig,
    key_cols_for_parent: Sequence[str],   # e.g., schema.all_cols or a subset incl. rot centers + angle
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates:
      1) per-aug compute limits + weights
      2) allocate slots across aug types
      3) sample indices & params for each type with dedicated RNGs
      4) materialize aug rows (bbox ints w/ rounding) + attach lineage
      5) merge next to parents deterministically
    Returns: (aug_df, file_index_merged, metadata_merged)
    """
    N = len(metadata_sorted)
    assert len(file_index_sorted) == N

    # Ensure bbox hygiene up front using your accessor (optional but recommended)
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
    # record how many per type and the order of slots per type
    per_type_indices = [np.where(slot_types == k)[0] for k in range(len(augmentations))]

    # 3) sample src indices & params per type with isolated RNGs
    # Create independent but deterministic seed streams from master seed
    # This gives us decorrelated but fully reproducible random streams
    all_generators = spawn_generators(2 * len(augmentations), config.seed)
    g_idx = all_generators[:len(augmentations)]  # First half for index sampling
    g_par = all_generators[len(augmentations):]  # Second half for parameter sampling
    aug_rows = []
    lineage_cols_common = [
        "_source_kind","_aug_type","_aug_slot","_parent_src_index","_parent_key",
        "_sampled_angle","_sampled_dx","_sampled_dy"
    ]  # present but some may be NaN

    parent_key_series = make_parent_key(
        file_index_sorted, metadata_sorted, key_cols_for_parent
    )
    parent_key_arr = parent_key_series.to_numpy()

    for k, aug in enumerate(augmentations):
        slot_ids = per_type_indices[k]
        if len(slot_ids) == 0:
            continue
        n_k = len(slot_ids)
        p = prob_list[k]
        limits = limits_list[k]

        # choose sources
        src_idx = weighted_choice(g_idx[k], p, size=n_k, replace=config.replace_per_aug)

        # Get limits for selected sources
        sel_limits = limits.iloc[src_idx].reset_index(drop=True)
        
        # DECORRELATION: Shuffle the parameter sampling order independently
        # This breaks the systematic relationship between source selection order and parameter values
        # while still respecting each source's individual constraints
        param_order = np.arange(n_k)
        g_par[k].shuffle(param_order)  # Shuffle in-place deterministically
        
        # Sample parameters in shuffled order, then reorder back to match sources
        sel_limits_shuffled = sel_limits.iloc[param_order].reset_index(drop=True)
        params_shuffled = aug.sample_params(g_par[k], sel_limits_shuffled)
        
        # Reorder parameters back to match original source order
        params = {}
        for key, values in params_shuffled.items():
            reorder_idx = np.argsort(param_order)  # Inverse permutation
            params[key] = values[reorder_idx]

        # materialize
        src_rows = metadata_sorted.iloc[src_idx].reset_index(drop=True)
        augs_k = aug.materialize(schema, src_rows, params, rounding=config.rounding)

        # attach lineage
        augs_k["_source_kind"] = "augmentation"
        augs_k["_aug_type"] = aug.name
        # within-parent stable order: rank by encounter
        # we’ll fill _aug_slot after concatenation across all augs
        augs_k["_aug_slot"] = np.nan
        augs_k["_parent_src_index"] = src_idx
        augs_k["_parent_key"] = parent_key_arr[src_idx]
        # ensure the optional metadata keys exist (NaN when unused)
        for col in ("_sampled_angle","_sampled_dx","_sampled_dy"):
            if col not in augs_k.columns:
                augs_k[col] = np.nan
        aug_rows.append(augs_k)

    # All augmentations together
    aug_df = pd.concat(aug_rows, axis=0, ignore_index=True) if aug_rows else pd.DataFrame(columns=metadata_sorted.columns.tolist() + lineage_cols_common)

    # assign _aug_slot deterministically per parent by order in aug_df
    if not aug_df.empty:
        aug_df["_tmp_row"] = np.arange(len(aug_df))
        aug_df["_aug_slot"] = (
            aug_df["_tmp_row"].groupby(aug_df["_parent_src_index"]).rank(method="first").astype(int) - 1
        )
        aug_df = aug_df.drop(columns=["_tmp_row"])

    # Originals with lineage
    originals = metadata_sorted.copy()
    originals["_source_kind"] = "original"
    originals["_aug_type"] = np.nan
    originals["_aug_slot"] = np.nan
    originals["_parent_src_index"] = np.arange(N, dtype=int)
    originals["_parent_key"] = parent_key_arr
    for col in ("_sampled_angle","_sampled_dx","_sampled_dy"):
        originals[col] = np.nan

    # 5) Merge & order by parent -> original first -> aug order
    if not aug_df.empty:
        file_index_augs = file_index_sorted.iloc[aug_df["_parent_src_index"].to_numpy()].reset_index(drop=True)
        file_index_augs.columns = file_index_sorted.columns
        file_index_merged = pd.concat([file_index_sorted.reset_index(drop=True), file_index_augs],
                                      axis=0, ignore_index=True)
        metadata_merged = pd.concat([originals.reset_index(drop=True), aug_df.reset_index(drop=True)],
                                    axis=0, ignore_index=True)

        helper = (
            metadata_merged.assign(
                __kind_rank = metadata_merged["_source_kind"].map({"original":0,"augmentation":1}).astype(int),
                __slot_rank = metadata_merged["_aug_slot"].fillna(-1).astype(int)
            )
        )
        order = helper.sort_values(["_parent_key","__kind_rank","__slot_rank"], kind="mergesort").index
        file_index_merged = file_index_merged.loc[order].reset_index(drop=True)
        metadata_merged   = metadata_merged.loc[order].reset_index(drop=True)
    else:
        # no augs
        file_index_merged = file_index_sorted.reset_index(drop=True)
        metadata_merged   = originals.reset_index(drop=True)

    # parent merged row index
    parent_pos = (
        metadata_merged.reset_index().groupby("_parent_key")["index"].min().rename("_parent_merged_index")
    )
    metadata_merged = metadata_merged.merge(parent_pos, left_on="_parent_key", right_index=True, how="left")

    return aug_df, file_index_merged, metadata_merged

"""
"""
def normalize_pathlike(x: pathlib.Path, case_sensitive: bool = True) -> str:
    return x.as_posix().lower() if not case_sensitive else x.as_posix()

def deterministic_sort_metadata(
    file_index: pd.DataFrame,
    metadata: pd.DataFrame,
    id_cols: Sequence[str],
    case_sensitive: bool = True,
    na_position: str = "last"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sorts the file index and metadata DataFrames in a deterministic way based on
    the file index plus additional identifying columns in metadata.

    :param file_index: DataFrame containing file paths or identifiers.
    :param metadata: DataFrame containing metadata associated with the file index.
    :param id_cols: List of column names in metadata that uniquely
        identify each image stack and its patches.
    :param case_sensitive: Whether to consider case sensitivity when normalizing paths.
    :param na_position: Position of NaN values in the sorted output ('first' or 'last').
    :return: Tuple of sorted file index and metadata DataFrames.
    """
    if len(file_index) != len(metadata):
        raise ValueError("file_index and metadata must have the same length")
    
    # build temporary dataframe with file index whose rows uniquely identify
    # image stacks plus patches coordinates that uniquely identify patches
    # within the site/view represented by each image stack
    file_index_norm = file_index.copy().map(
        lambda p: normalize_pathlike(p, case_sensitive=case_sensitive) \
            if isinstance(p, pathlib.Path) else p
    )
    key_df = pd.concat(
        [file_index_norm, 
         metadata.loc[:, id_cols]], axis=1)
    
    # perform stable lexicographic sort across all key columns
    order = key_df.sort_values(list(key_df.columns),
                               kind="mergesort",    # stable
                               na_position=na_position).index
    
    file_index_sorted = file_index.loc[order].reset_index(drop=True)
    metadata_sorted = metadata.loc[order].reset_index(drop=True)

    return file_index_sorted, metadata_sorted

"""
"""
def compute_safe_translation_limits(
    metadata: pd.DataFrame,
    fov_h: Union[int, np.ndarray],
    fov_w: Union[int, np.ndarray],
    bbox_col_prefix: str = '_patch_bbox_',
    max_translation_x: Optional[float] = None,
    max_translation_y: Optional[float] = None,
):
    
    bbox_df = metadata.loc[
        :,
        [f'{bbox_col_prefix}{col}' for col in [
            'x_min', 'y_min', 'x_max', 'y_max']]
    ]
    bbox_df.columns = [
        'x_min', 'y_min', 'x_max', 'y_max'
    ]
    
    safe_offset = pd.DataFrame(
        columns=['dx_min', 'dx_max', 'dy_min', 'dy_max'],
        index=bbox_df.index
    )

    # expand fov_h and fov_w to match bbox_df length if they are single values
    if isinstance(fov_h, int):
        fov_h = np.full(len(bbox_df), fov_h, dtype=int)
    if isinstance(fov_w, int):
        fov_w = np.full(len(bbox_df), fov_w, dtype=int)

    if max_translation_x is None:
        max_translation_x = (bbox_df['x_max'] - bbox_df['x_min']) * 0.45
    if max_translation_y is None:
        max_translation_y = (bbox_df['y_max'] - bbox_df['y_min']) * 0.45
    
    safe_offset['dx_min'] = np.maximum(- 1 * bbox_df['x_min'], -1 * max_translation_x)
    safe_offset['dx_max'] = np.minimum(fov_w - bbox_df['x_max'], max_translation_x)
    safe_offset['dy_min'] = np.maximum(-1 * bbox_df['y_min'], -1 * max_translation_y)
    safe_offset['dy_max'] = np.minimum(fov_h - bbox_df['y_max'], max_translation_y)

    return safe_offset

def compute_safe_rotation_limits(
    metadata: pd.DataFrame,
    fov_h: Union[int, np.ndarray],
    fov_w: Union[int, np.ndarray],
    step_deg: float = 1.0,
    bbox_col_prefix: str = '_patch_bbox_',
) -> pd.DataFrame:
    """
    Compute safe rotation limits for bounding boxes.
    Optimized version using vectorized operations.
    """
    bbox_df = metadata.loc[
        :,
        [f'{bbox_col_prefix}{col}' for col in [
            'x_min', 'y_min', 'x_max', 'y_max', 'rot_x_center', 'rot_y_center']]
    ]
    bbox_df.columns = ['x_min', 'y_min', 'x_max', 'y_max', 'cx', 'cy']
    
    fov_h = ensure_array_broadcast(fov_h, len(bbox_df))
    fov_w = ensure_array_broadcast(fov_w, len(bbox_df))
    
    limits_data = []
    for i, row in bbox_df.iterrows():
        bbox = (row['x_min'], row['y_min'], row['x_max'], row['y_max'])
        center = (row['cx'], row['cy'])
        fov_shape = (fov_h[i], fov_w[i])
        
        min_angle, max_angle = find_max_rotation_range(bbox, center, fov_shape, step_deg)
        limits_data.append([min_angle, max_angle])
    
    return pd.DataFrame(limits_data, columns=['min_angle', 'max_angle'])