"""
/datasets/run_aug.py

This file implements the AugRunner class and infranstructure for reproducible
augmentation of bounding box datasets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict, Any
import pathlib

import pandas as pd
import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

from .bbox_schema import BBoxSchema
from .real_augments import BaseAugmentation

@dataclass
class AugPlanConfig:
    seed: int = 42
    rounding: str = "round"
    replace_per_aug: bool = True
    weights_per_aug: Optional[List[float]] = None
    # Sorting ON by default
    sort_inputs: bool = True
    case_sensitive_paths: bool = True
    # this is for the final sorting of merged augmetnation
    # due to different kind of augmentations having different parameters,
    # and the augmentation df will be the union of all these parameter columns
    # with blocks of NaNs for the augmentations that do not have these parameters
    # this parameter defines how to sort these NaNs, by default they are sorted last
    na_position: str = "last" 

def spawn_generators(n: int, seed: int) -> List[Generator]:
    ss = SeedSequence(seed)
    return [Generator(PCG64(s)) for s in ss.spawn(n)]

def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p[~np.isfinite(p)] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))

def normalize_pathlike(p: Union[str, pathlib.Path], case_sensitive: bool=True) -> str:
    s = str(p)
    return s if case_sensitive else s.lower()

def weighted_choice(gen: np.random.Generator, p: np.ndarray, size: int, replace: bool=True) -> np.ndarray:
    p = normalize_probs(p)
    idx = gen.choice(len(p), size=size, replace=replace, p=p)
    return idx.astype(np.int64)

def make_parent_key_default(file_index_df: pd.DataFrame, bbox_df: pd.DataFrame) -> pd.Series:
    # Deterministic key from full file_index row + bbox row index
    finfo = file_index_df.astype(str)
    fi_key = finfo.astype(str).agg("||".join, axis=1)
    return fi_key + ("||bboxidx=" + bbox_df.index.astype(str))

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

class AugRunner:
    """
    Schema-agnostic augmenter runner operating on:
      - file_index: pd.DataFrame (paths/IDs)
      - bbox_df:    pd.DataFrame (already schema-registered with all bbox cols)

    Augmentation interface (duck-typed):
      - .name: str
      - .compute_limits(bbox_df, schema) -> (limits_df, sampling_probs)
      - .sample_params(gen_or_none, sel_limits, keyed_info) -> params_df/obj
      - .materialize(schema, source_rows, params, rounding) -> augmented_rows_df (same bbox columns)
    """

    LINEAGE_COLS = [
        "_source_kind","_aug_type","_aug_slot","_parent_src_index","_parent_key",
        "_sampled_angle","_sampled_dx","_sampled_dy"
    ]

    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        config: AugPlanConfig,
        schema: BBoxSchema,
        make_parent_key_fn = make_parent_key_default,
    ):
        self.augs = augmentations
        self.cfg = config
        self.schema = schema
        self.make_parent_key_fn = make_parent_key_fn

    # ---------- sorting helpers ----------
    def _compute_order(self, file_index: pd.DataFrame, bbox_df: pd.DataFrame) -> pd.Index:
        # Build a combined key DataFrame from normalized file_index + bbox index
        finfo = file_index.copy().map(
            lambda p: normalize_pathlike(p, case_sensitive=self.cfg.case_sensitive_paths)
            if isinstance(p, (str, pathlib.Path)) else str(p)
        )
        key_df = finfo.copy()
        key_df["__bbox_index__"] = bbox_df.index.astype(str)
        order = key_df.sort_values(
            list(key_df.columns),
            kind="mergesort",
            na_position=self.cfg.na_position
        ).index
        return order

    def _apply_order_to_payloads(
        self,
        order: pd.Index,
        payloads: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Reorder any payload aligned to originals.
        Accepts mixtures of:
          - list/tuple/np.ndarray (length N)
          - pd.Series / pd.DataFrame (length N)
        Returns a new dict with re-ordered payloads (same keys).
        """
        if payloads is None:
            return None
        N = len(order)
        out: Dict[str, Any] = {}
        for k, v in payloads.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                if len(v) != N:
                    raise ValueError(f"Payload '{k}' length {len(v)} != originals length {N}")
                out[k] = v.iloc[order].reset_index(drop=True)
            else:
                arr = np.asarray(v, dtype=object)  # dtype=object to be permissive
                if len(arr) != N:
                    raise ValueError(f"Payload '{k}' length {len(arr)} != originals length {N}")
                out[k] = arr[order]
        return out

    def _maybe_sort_inputs(
        self,
        file_index: pd.DataFrame,
        bbox_df: pd.DataFrame,
        payloads: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]], pd.Index]:
        if not self.cfg.sort_inputs:
            order = pd.RangeIndex(len(bbox_df))
            return (file_index.reset_index(drop=True),
                    bbox_df.reset_index(drop=True),
                    payloads,
                    order)
        order = self._compute_order(file_index, bbox_df)
        file_index_sorted = file_index.loc[order].reset_index(drop=True)
        bbox_sorted = bbox_df.loc[order].reset_index(drop=True)
        payloads_sorted = self._apply_order_to_payloads(order, payloads)
        return file_index_sorted, bbox_sorted, payloads_sorted, order

    # ---------- main entry point ----------
    def run(
        self,
        file_index: pd.DataFrame,
        bbox_df: pd.DataFrame,
        n_slots: int,
        *,
        weights_per_aug: Optional[List[float]] = None,
        payloads: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Returns:
          aug_df, file_index_merged, bbox_merged, payloads_sorted

        Notes:
          - payloads (optional): dict of list/array/Series/DataFrame aligned to ORIGINAL rows.
            They are deterministically re-ordered to match sorted inputs and returned as payloads_sorted.
        """
        (
            file_index_sorted, bbox_sorted, payloads_sorted, _
            ) = self._maybe_sort_inputs(
            file_index, bbox_df, payloads
        )
        N = len(bbox_sorted)
        if len(file_index_sorted) != N:
            raise ValueError("file_index and bbox_df must have the same number of rows")

        # Produce keys to uniquely identify each original input bounding box
        # so the augmentations produced can be back-traced to their source.
        parent_key = self.make_parent_key_fn(file_index_sorted, bbox_sorted)
        parent_key_arr = parent_key.to_numpy()

        # 1) per-aug limits + sampling probs
        limits_list: List[pd.DataFrame] = []
        prob_list: List[np.ndarray] = []
        for aug in self.augs:
            limits, p = aug.compute_limits(bbox_sorted, self.schema)
            limits_list.append(limits.reset_index(drop=True))
            prob_list.append(np.asarray(p, dtype=float))

        # 2) deterministic slot allocation
        aug_names = [a.name for a in self.augs]
        slot_types = allocate_augmentation_slots(
            n_slots,
            aug_names,
            weights=(weights_per_aug if (weights_per_aug is not None) else self.cfg.weights_per_aug),
            seed=self.cfg.seed
        )
        per_type_indices = [np.where(slot_types == k)[0] for k in range(len(self.augs))]

        # 3) keyed RNGs per aug type (for source index sampling)
        g_idx = spawn_generators(len(self.augs), self.cfg.seed)

        aug_rows = []
        for k, aug in enumerate(self.augs):
            slot_ids = per_type_indices[k]
            if len(slot_ids) == 0:
                continue

            n_k = len(slot_ids)
            p = prob_list[k]
            limits = limits_list[k]

            # choose sources reproducibly
            src_idx = weighted_choice(g_idx[k], p, size=n_k, replace=self.cfg.replace_per_aug)
            sel_limits = limits.iloc[src_idx].reset_index(drop=True)

            # occurrence ranks (keyed param seeding)
            occ_ranks = _occurrence_ranks(src_idx)

            keyed_info = {
                "base_seed": int(self.cfg.seed),
                "aug_index": int(k),
                "parent_indices": src_idx.astype(np.int64),
                "occurrence_ranks": occ_ranks.astype(np.int64),
            }
            params = aug.sample_params(
                limits_sel=sel_limits,
                gen=None,
                keyed_info=keyed_info
            )

            # materialize
            src_rows = bbox_sorted.iloc[src_idx].reset_index(drop=True)
            augs_k = aug.materialize(
                self.schema, src_rows, params, rounding=self.cfg.rounding)

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

        # 4) concatenate aug rows
        if aug_rows:
            aug_df = pd.concat(aug_rows, axis=0, ignore_index=True)
            aug_df["_tmp_row"] = np.arange(len(aug_df))
            aug_df["_aug_slot"] = (
                aug_df["_tmp_row"].groupby(aug_df["_parent_src_index"]).rank(method="first").astype(int) - 1
            )
            aug_df = aug_df.drop(columns=["_tmp_row"])
        else:
            aug_df = pd.DataFrame(columns=list(bbox_sorted.columns) + self.LINEAGE_COLS)

        # 5) originals with lineage
        originals = bbox_sorted.copy()
        originals["_source_kind"] = "original"
        originals["_aug_type"] = np.nan
        originals["_aug_slot"] = np.nan
        originals["_parent_src_index"] = np.arange(N, dtype=int)
        originals["_parent_key"] = parent_key_arr
        for col in ("_sampled_angle","_sampled_dx","_sampled_dy"):
            originals[col] = np.nan

        # 6) merge/sort originals + augs, keep file_index in lockstep
        if not aug_df.empty:
            file_index_augs = file_index_sorted.iloc[aug_df["_parent_src_index"].to_numpy()].reset_index(drop=True)
            file_index_augs.columns = file_index_sorted.columns

            file_index_merged = pd.concat(
                [file_index_sorted.reset_index(drop=True), file_index_augs],
                axis=0, ignore_index=True
            )
            bbox_merged = pd.concat(
                [originals.reset_index(drop=True), aug_df.reset_index(drop=True)],
                axis=0, ignore_index=True
            )

            helper = bbox_merged.assign(
                __kind_rank=bbox_merged["_source_kind"].map({"original":0, "augmentation":1}).astype(int),
                __slot_rank=bbox_merged["_aug_slot"].fillna(-1).astype(int)
            )
            order2 = helper.sort_values(
                ["_parent_key","__kind_rank","__slot_rank"],
                kind="mergesort"
            ).index

            file_index_merged = file_index_merged.loc[order2].reset_index(drop=True)
            bbox_merged = bbox_merged.loc[order2].reset_index(drop=True)
        else:
            file_index_merged = file_index_sorted.reset_index(drop=True)
            bbox_merged = originals.reset_index(drop=True)

        # 7) parent merged index for grouping/formatting
        parent_pos = (
            bbox_merged.reset_index().groupby("_parent_key")["index"].min().rename("_parent_merged_index")
        )
        bbox_merged = bbox_merged.merge(parent_pos, left_on="_parent_key", right_index=True, how="left")

        gather_idx = bbox_merged["_parent_src_index"].to_numpy()
        payloads_expanded = self._expand_payloads(payloads_sorted, gather_idx)

        return aug_df, file_index_merged, bbox_merged, payloads_expanded
    
    def _expand_payloads(
        self,
        payloads_sorted: Optional[Dict[str, Any]],
        gather_idx: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Expand payloads aligned to ORIGINAL rows (length N) to MERGED length (L)
        by copying each original's value to all of its children.
        No dtype changes; returns the same types as provided:
        - pd.Series -> pd.Series
        - pd.DataFrame -> pd.DataFrame
        - np.ndarray -> np.ndarray
        - list/tuple -> list
        """
        if payloads_sorted is None:
            return None

        out: Dict[str, Any] = {}
        for k, v in payloads_sorted.items():
            if isinstance(v, pd.DataFrame):
                expanded = v.iloc[gather_idx].reset_index(drop=True)
            elif isinstance(v, pd.Series):
                expanded = v.iloc[gather_idx].reset_index(drop=True)
            else:
                arr = np.asarray(v)
                expanded_arr = arr[gather_idx]
                # preserve type for ndarray; convert lists/tuples back to list
                if isinstance(v, np.ndarray):
                    expanded = expanded_arr
                elif isinstance(v, (list, tuple)):
                    expanded = expanded_arr.tolist()
                else:
                    # fallback: keep as ndarray
                    expanded = expanded_arr
            out[k] = expanded
        return out