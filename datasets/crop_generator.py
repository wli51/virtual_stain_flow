from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

from .bbox_schema import BBoxSchema

class CropGenerator(ABC):
    """Abstract base class for crop generation strategies."""
    
    @abstractmethod
    def generate_crops(
        self,
        file_index: pd.DataFrame,
        object_metadata: List[pd.DataFrame],
        **kwargs
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Generate crops from object metadata.
        
        :param file_index: DataFrame with file paths
        :param object_metadata: List of DataFrames with object coordinates
        :return: Tuple of (bboxes_df_list, filtered_obj_metadata_list)
        """
        pass
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Return list of required parameter names."""
        pass

class ObjectCenteredCropGenerator(CropGenerator):
    """Generate square crops centered on objects with filtering."""
    
    def __init__(
        self,
        crop_size: int,
        object_coord_x_field: str,
        object_coord_y_field: str,
        fov: Optional[Tuple[int, int]] = None,
        bbox_schema: Optional[BBoxSchema] = None,
        apply_deduplication: bool = True,
        apply_subset_filtering: bool = True,
        apply_minimal_selection: bool = True
    ):
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError("crop_size must be a positive integer.")
        if not isinstance(object_coord_x_field, str):
            raise TypeError("object_coord_x_field must be a string.")
        if not isinstance(object_coord_y_field, str):
            raise TypeError("object_coord_y_field must be a string.")
            
        self.crop_size = crop_size
        self.object_coord_x_field = object_coord_x_field
        self.object_coord_y_field = object_coord_y_field
        self.fov = fov
        self.bbox_schema = bbox_schema
        self.apply_deduplication = apply_deduplication
        self.apply_subset_filtering = apply_subset_filtering
        self.apply_minimal_selection = apply_minimal_selection
    
    def generate_crops(
        self,
        file_index: pd.DataFrame,
        object_metadata: List[pd.DataFrame],
        **kwargs
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Generate object-centered crops with filtering.
        
        :param file_index: DataFrame containing file paths
        :param object_metadata: List of DataFrames with object coordinates
        :return: Tuple of (bboxes_df_list, filtered_obj_metadata_list)
        """
        bboxes_df_list = [None] * len(file_index)
        bboxes_obj_metadata_list = []
        rows = list(file_index.iterrows())

        for (i, file_index_row), obj_meta_df in tqdm(
            zip(rows, object_metadata),
            total=len(rows),
            desc='Generating object-centered crops...'
        ):
            # Validate required columns
            self._validate_coord_columns(obj_meta_df, i)
            
            # Extract and normalize coordinates
            coords_df = self._extract_coordinates(obj_meta_df)
            
            # Get FOV shape
            fov_shape = self._get_fov_shape(file_index_row)
            
            # Generate bboxes centered on objects
            bboxes_df = self._generate_obj_centered_bboxes(coords_df, fov_shape)
            
            # Apply filtering pipeline
            selected_bboxes, selected_obj_indices = self._apply_filtering_pipeline(
                bboxes_df, coords_df, obj_meta_df
            )
            
            # Store results
            bboxes_df_list[i] = selected_bboxes
            bboxes_obj_metadata_list.extend([
                obj_meta_df.iloc[[j]].reset_index(drop=True) 
                for j in selected_obj_indices
            ])

        return bboxes_df_list, bboxes_obj_metadata_list
    
    def get_required_params(self) -> List[str]:
        return ['crop_size', 'object_coord_x_field', 'object_coord_y_field']
    
    def _validate_coord_columns(self, obj_meta_df: pd.DataFrame, index: int):
        """Validate that required coordinate columns exist."""
        missing_cols = []
        if self.object_coord_x_field not in obj_meta_df.columns:
            missing_cols.append(self.object_coord_x_field)
        if self.object_coord_y_field not in obj_meta_df.columns:
            missing_cols.append(self.object_coord_y_field)
            
        if missing_cols:
            raise ValueError(
                f"Missing columns {missing_cols} in object metadata entry {index+1}."
            )
    
    def _extract_coordinates(self, obj_meta_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize coordinate columns."""
        coords_df = obj_meta_df[[self.object_coord_x_field, self.object_coord_y_field]].copy()
        coords_df.columns = ['x', 'y']
        coords_df.reset_index(drop=True, inplace=True)
        return coords_df
    
    def _get_fov_shape(self, file_index_row: pd.Series) -> Tuple[int, int]:
        """Get FOV shape, either from provided fov or by inferring from file."""
        if self.fov is not None:
            return self.fov
        else:
            return self._infer_fov(file_index_row.iloc[0])
    
    def _infer_fov(self, img_file: Path) -> Tuple[int, int]:
        """Infer FOV dimensions from image file."""
        try:
            from PIL import Image
            with Image.open(img_file) as img:
                fov_w, fov_h = img.size  # PIL returns (width, height)
                return fov_h, fov_w  # Return as (height, width)
        except Exception as e:
            raise RuntimeError(f"Failed to open image file {img_file}: {e}")
    
    def _generate_obj_centered_bboxes(
        self, 
        coords_df: pd.DataFrame, 
        fov_shape: Tuple[int, int]
    ) -> pd.DataFrame:
        """Generate square bounding boxes centered around object coordinates."""
        half = self.crop_size // 2
        fov_h, fov_w = fov_shape

        # Compute unclamped corners
        x_min = coords_df['x'] - half
        y_min = coords_df['y'] - half
        x_max = coords_df['x'] + half
        y_max = coords_df['y'] + half

        # Clamp to FOV boundaries
        x_min = np.clip(x_min, 0, fov_w - self.crop_size)
        y_min = np.clip(y_min, 0, fov_h - self.crop_size)
        x_max = x_min + self.crop_size
        y_max = y_min + self.crop_size

        # Create DataFrame with appropriate column names
        if self.bbox_schema is not None:
            columns = {
                'x_min': self.bbox_schema.x_min,
                'y_min': self.bbox_schema.y_min,
                'x_max': self.bbox_schema.x_max,
                'y_max': self.bbox_schema.y_max,
            }
        else:
            columns = {
                'x_min': 'x_min',
                'y_min': 'y_min', 
                'x_max': 'x_max',
                'y_max': 'y_max'
            }

        df = pd.DataFrame({
            columns['x_min']: x_min.astype(int),
            columns['y_min']: y_min.astype(int),
            columns['x_max']: x_max.astype(int),
            columns['y_max']: y_max.astype(int)
        })

        return df
    
    def _apply_filtering_pipeline(
        self, 
        bboxes_df: pd.DataFrame, 
        coords_df: pd.DataFrame,
        obj_meta_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Apply the complete filtering pipeline to bboxes."""
        
        # Compute membership matrix
        membership_matrix = self._compute_bboxes_obj_membership(bboxes_df, coords_df)
        
        # Start with all indices
        indices = list(range(len(bboxes_df)))
        
        # Apply filtering steps
        if self.apply_deduplication:
            indices = self._deduplicate_bboxes(membership_matrix, indices)
            
        if self.apply_subset_filtering:
            indices = self._drop_subset_bboxes(membership_matrix, indices)
            
        if self.apply_minimal_selection:
            indices = self._select_minimal_bboxes(membership_matrix, indices)
        
        # Return filtered bboxes and corresponding object indices
        selected_bboxes = bboxes_df.iloc[indices].reset_index(drop=True)
        
        return selected_bboxes, indices
    
    def _compute_bboxes_obj_membership(
        self, 
        bboxes_df: pd.DataFrame,
        coords_df: pd.DataFrame
    ) -> np.ndarray:
        """Compute membership matrix indicating which objects fall within each bbox."""
        
        # Get column names based on schema
        if self.bbox_schema is None:
            x_min_col, y_min_col = 'x_min', 'y_min'
            x_max_col, y_max_col = 'x_max', 'y_max'
        else:
            x_min_col = self.bbox_schema.x_min
            y_min_col = self.bbox_schema.y_min
            x_max_col = self.bbox_schema.x_max
            y_max_col = self.bbox_schema.y_max

        # Extract bbox coordinates
        x_min = bboxes_df[x_min_col].to_numpy()
        y_min = bboxes_df[y_min_col].to_numpy()
        x_max = bboxes_df[x_max_col].to_numpy()
        y_max = bboxes_df[y_max_col].to_numpy()

        # Extract object coordinates
        xs = coords_df['x'].to_numpy()
        ys = coords_df['y'].to_numpy()

        # Vectorized membership computation
        x_in = (x_min[:, None] <= xs[None, :]) & (xs[None, :] <= x_max[:, None])
        y_in = (y_min[:, None] <= ys[None, :]) & (ys[None, :] <= y_max[:, None])

        return x_in & y_in
    
    def _deduplicate_bboxes(
        self, 
        membership_matrix: np.ndarray, 
        indices: List[int]
    ) -> List[int]:
        """Remove duplicate bounding boxes based on membership masks."""
        
        sub_matrix = membership_matrix[indices, :]
        seen = {}
        unique_indices = []

        for i, mask in enumerate(sub_matrix):
            key = mask.tobytes()
            if key not in seen:
                seen[key] = i
                unique_indices.append(indices[i])

        return unique_indices
    
    def _drop_subset_bboxes(
        self, 
        membership_matrix: np.ndarray, 
        indices: List[int]
    ) -> List[int]:
        """Drop subset bounding boxes based on membership masks."""
        
        sub_matrix = membership_matrix[indices, :]
        sizes = sub_matrix.sum(axis=1)
        sorted_order = np.argsort(-sizes)

        kept_indices = []
        kept_masks = []

        for idx in sorted_order:
            mask = sub_matrix[idx]
            if not any(np.all(mask <= kmask) for kmask in kept_masks):
                kept_indices.append(indices[idx])
                kept_masks.append(mask)

        return kept_indices
    
    def _select_minimal_bboxes(
        self, 
        membership_matrix: np.ndarray, 
        indices: List[int]
    ) -> List[int]:
        """Select minimal set of bboxes that cover all objects."""
        
        sub_matrix = membership_matrix[indices, :]
        num_objects = sub_matrix.shape[1]
        covered = np.zeros(num_objects, dtype=bool)
        selected_indices = []
        remaining = set(range(len(indices)))

        while not np.all(covered) and remaining:
            best_box = None
            best_gain = -1

            for i in remaining:
                gain = np.sum(sub_matrix[i] & ~covered)
                if gain > best_gain:
                    best_gain = gain
                    best_box = i

            if best_gain == 0:
                break

            selected_indices.append(indices[best_box])
            covered |= sub_matrix[best_box]
            remaining.remove(best_box)

        return selected_indices