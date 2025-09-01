"""
datasets/bbox_schema.py

This file contains the schema/view/accessor definition to 
extend the BaseImageDataset class .metadata DataFrame attribute as
a structured representation of bounding box annotations.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import pandas as pd

"""
Centralized definition of bbox-related column names.
Everything derives from a prefix, so you never spread strings around.
"""
@dataclass(frozen=True)
class BBoxSchema:
    
    version: str = '0.1'
    prefix: str = "_patch_bbox_" # TODO tentative prefix

    # Base column keys
    _keys = (
        # Standard bbox coordinates
        "x_min", "y_min", "x_max", "y_max",
        # Center of mass coordinate that will
        # be auto-computed from the coordinates
        "box_x_center", "box_y_center",
        # Rotation center of the box, if not
        # explicitly defined, will be
        # the same as the center of mass
        "rot_x_center", "rot_y_center",
        # Angle of rotation in degrees, 0 means no rotation
        "angle",
    )

    def col(self, key: str) -> str:
        if key not in self._keys:
            raise KeyError(f"Unknown bbox key: {key}")
        return f"{self.prefix}{key}"
    
    # defines boxes with standard min/max coordinates 
    @property
    def x_min(self) -> str: return self.col("x_min")
    @property
    def y_min(self) -> str: return self.col("y_min")
    @property
    def x_max(self) -> str: return self.col("x_max")
    @property
    def y_max(self) -> str: return self.col("y_max")

    # these are for the center of mass of the box
    @property
    def cx(self) -> str: return self.col("box_x_center")
    @property
    def cy(self) -> str: return self.col("box_y_center")

    # these are for the rotation center of the box, if applicable/different
    # from the center of mass
    @property
    def rot_cx(self) -> str: return self.col("rot_x_center")
    @property
    def rot_cy(self) -> str: return self.col("rot_y_center")

    # angle of rotation in degrees, 0 means no rotation
    @property
    def angle(self) -> str: return self.col("angle")

    # Handy bundles
    @property
    def bbox_cols(self) -> Tuple[str, str, str, str]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def center_cols(self) -> Tuple[str, str]:
        return (self.cx, self.cy)

    @property
    def rot_center_cols(self) -> Tuple[str, str]:
        return (self.rot_cx, self.rot_cy)

    @property
    def all_cols(self) -> Tuple[str, ...]:
        return tuple(self.col(k) for k in self._keys)
    
    # ---------- serialization ----------
    def to_dict(self) -> dict:
        return {"prefix": self.prefix}
    
    @classmethod
    def from_dict(cls, d: dict) -> "BBoxSchema":
        # tolerate extra fields; require prefix; default version
        prefix = d.get("prefix")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("BBoxSchema.from_dict: 'prefix' must be a non-empty string.")
        return cls(prefix=prefix)
    
"""
Wraps a pandas Series (row) and a BBoxSchema to provide clean attribute access
when iterating over rows.
"""
class BBoxRowView:
    
    def __init__(self, row: pd.Series, schema: BBoxSchema):
        self._row = row
        self._s = schema

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        r = self._row
        return int(r[self._s.x_min]), int(r[self._s.y_min]), int(r[self._s.x_max]), int(r[self._s.y_max])

    @property
    def center(self) -> Tuple[float, float]:
        r = self._row
        return float(r[self._s.cx]), float(r[self._s.cy])

    @property
    def rot_center(self) -> Tuple[float, float]:
        r = self._row
        return float(r[self._s.rot_cx]), float(r[self._s.rot_cy])

    @property
    def angle(self) -> float:
        return float(self._row[self._s.angle])

"""
pandas Accessor
"""
@pd.api.extensions.register_dataframe_accessor("bbox")
class BBoxAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._df = pandas_obj
        self._schema = BBoxSchema()  # default; set via __call__

    def __call__(self, schema: BBoxSchema) -> "BBoxAccessor":
        acc = BBoxAccessor(self._df)
        acc._schema = schema
        return acc

    # --- Checks if a metadata dataframe contains valid bbox annotations ---
    def ensure_columns(self) -> pd.DataFrame:
        s, df = self._schema, self._df
        for c in s.bbox_cols:
            if c not in df.columns:
                raise ValueError(f"Missing bbox column: {c}")
            df[c] = df[c].astype(int)

        # Compute center of mass if not present
        if s.cx not in df.columns or s.cy not in df.columns:
            df[s.cx] = (df[s.x_min] + df[s.x_max]) / 2.0
            df[s.cy] = (df[s.y_min] + df[s.y_max]) / 2.0

        # Assign rotation center if not present
        if s.rot_cx not in df.columns:
            df[s.rot_cx] = df[s.cx]
        if s.rot_cy not in df.columns:
            df[s.rot_cy] = df[s.cy]

        df[s.rot_cx] = df[s.rot_cx].astype(float)
        df[s.rot_cy] = df[s.rot_cy].astype(float)

        if s.angle not in df.columns:
            df[s.angle] = 0.0
        df[s.angle] = df[s.angle].astype(float)
        return df
    
    def _map_keys(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        return [getattr(self._schema, k) if not k.startswith(self._schema.prefix) else k
                for k in keys]

    def cols(self, *keys: str) -> pd.DataFrame:
        """Return a DataFrame with the bbox columns specified by logical names."""
        return self._df[self._map_keys(keys)]

    def series(self, key: str) -> pd.Series:
        """Return a single bbox column as a Series by logical name."""
        return self._df[self._map_keys(key)[0]]

    def np(self, *keys: str, rows=None, dtype=None, order="C") -> np.ndarray:
        """
        Return selected bbox columns as a NumPy array.

        :param keys: logical names of the bbox columns
        :param rows: slice/list/boolean mask/None
        :param dtype: data type for the output array
        :param order: memory layout order ('C' or 'F')
        """
        df = self.cols(*keys)
        if rows is not None:
            df = df.loc[rows]
        arr = df.to_numpy(dtype=dtype, copy=False)
        return np.array(arr, dtype=dtype, order=order, copy=False)

    def select(self, rows, *keys: str) -> pd.DataFrame:
        """Row-and-column selection in one call."""
        return self._df.loc[rows, self._map_keys(keys)]

    # --- write helpers ---
    def set(self, key: str, values) -> pd.DataFrame:
        """Assign to a bbox column by logical name; returns the DataFrame."""
        self._df[self._map_keys(key)[0]] = values
        return self._df

    # --- sugar: bracket access with logical names ---
    def __getitem__(self, item):
        """
        df.bbox(schema)['x_min'] -> Series
        df.bbox(schema)[['x_min','y_min']] -> DataFrame
        """
        if isinstance(item, (list, tuple)):
            return self.cols(*item)
        return self.series(item)

    def row(self, i: int) -> BBoxRowView:
        return BBoxRowView(self._df.iloc[i], self._schema)

    def coords(self, i: int):
        return self.row(i).bbox

    def centers(self, i: int):
        return self.row(i).center

    def rot_centers(self, i: int):
        return self.row(i).rot_center

    def angle_of(self, i: int) -> float:
        return self.row(i).angle
