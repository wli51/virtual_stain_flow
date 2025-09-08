"""
bbox_dataset.py

Dataset class that supports cropping and optional rotation of images.
Requires user specified bounding box annotations as a DataFrame, plus
a matching file index DataFrame containing file paths to the raw images.

Class: 
- BBoxCropImageDataset: Inherits from BaseImageDataset, 
    overrides get_raw_item to apply cropping
    and rotation to the raw images before returning them.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .bbox_schema import BBoxSchema, BBoxAccessor
from .base_dataset import BaseImageDataset
from .image_utils import crop_and_rotate_image

class BBoxCropImageDataset(BaseImageDataset):
    def __init__(
        self,
        file_index: pd.DataFrame,
        bbox_annotations: pd.DataFrame,
        bbox_schema: BBoxSchema = BBoxSchema(),
        **kwargs
    ):
        """
        Initializes the BBoxCropImageDataset.

        :param file_index: DataFrame containing file paths to the raw images. 
        :param bbox_annotations: DataFrame containing bounding box annotations.
            Each row corresponds to a crop with bbox coordinates and optional rotation.
            Must match the length of file_index.
        :param bbox_schema: BBoxSchema object defining column names.
        :param kwargs: Additional arguments passed to BaseImageDataset.
        """
        self._schema = self._validate_schema(bbox_schema)
        self._bbox_df = self._prepare_bbox_dataframe(bbox_annotations)
        
        super().__init__(file_index=file_index, **kwargs)

    def _validate_schema(self, bbox_schema: BBoxSchema) -> BBoxSchema:
        """Validate bbox_schema type."""
        if not isinstance(bbox_schema, BBoxSchema):
            raise TypeError(f"Expected BBoxSchema, got {type(bbox_schema)}")
        return bbox_schema
    
    def _prepare_bbox_dataframe(self, bbox_annotations: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate bbox annotations DataFrame."""
        if not isinstance(bbox_annotations, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(bbox_annotations)}")
        
        # Create accessor and ensure required columns
        df_copy = bbox_annotations.copy()
        bbox_accessor = df_copy.bbox(self._schema)
        prepared_df = bbox_accessor.ensure_columns()
        
        # Store the accessor with column mappings for later use
        self._bbox: BBoxAccessor = bbox_accessor

        return prepared_df

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def get_raw_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cropped images for the given index.
        
        :param idx: Index of the item to retrieve.
        :return: Tuple of cropped numpy arrays (input_image, target_image).
        """
        # Get raw full images from parent
        raw_input, raw_target = super().get_raw_item(idx)

        # Get crop parameters using the new convenience method
        bbox = self._bbox.coords(idx)
        rcx, rcy = self._bbox.rot_centers(idx)
        angle = self._bbox.angle_of(idx)

        # Apply cropping and rotation to both images
        cropped_input = crop_and_rotate_image(raw_input, bbox, rcx, rcy, angle)
        cropped_target = crop_and_rotate_image(raw_target, bbox, rcx, rcy, angle)
        
        return cropped_input, cropped_target
    
    """
    These attributes allow for access of the schema and 
    DataFrame containing bounding box annotations, immutable
    after initialization.
    """
    @property
    def bbox_schema(self) -> BBoxSchema:
        """
        Returns the schema for bounding box annotations.
        """
        return self._schema
    @property
    def bbox_accessor(self):
        """
        Returns the accessor for bounding box annotations.
        This provides methods to access and manipulate bounding box data.
        """
        return self._bbox
    @property
    def bbox_df(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing bounding box annotations.
        """
        return self._bbox_df

    """
    Additional bbox specific state property
    """
    @property
    def bbox_state(self) -> Tuple[int, int, int, int, float, float, float]:
        """
        Returns the current bbox state as (xmin, ymin, xmax, ymax, rcx, rcy, angle).
        """
        last_idx = self.index_state.last
        if last_idx is None:
            raise ValueError("Index state not set. Cannot retrieve bbox state.")
        
        return (*self._bbox.coords(last_idx), 
                *self._bbox.rot_centers(last_idx), 
                self._bbox.angle_of(last_idx))

    """
    Serialization methods
    """
    def to_config(self) -> Dict[str, Any]:
        """
        Serialize configuration including bbox schema and annotations.

        :return: Dictionary containing the serialized configuration.
        """
        config = super().to_config()
        config.update({
            'bbox_schema': self.bbox_schema.to_dict(),
            'bbox_schema_class': f"{self.bbox_schema.__class__.__module__}.{self.bbox_schema.__class__.__name__}",
            'bbox_annotations': self.bbox_df.to_dict('records')
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'BBoxCropImageDataset':
        """
        Deserialize from configuration dictionary.
        
        :param config: Configuration dictionary
        :param kwargs: Additional keyword arguments
        :return: BBoxCropImageDataset instance
        """
        import importlib
        
        # Get base configuration
        core_kwargs = cls._deserialize_config_core(config)

        # Deserialize bbox annotations
        bbox_annotations_data = config.get('bbox_annotations')
        if not bbox_annotations_data:
            raise ValueError("Missing 'bbox_annotations' in config")
        bbox_annotations = pd.DataFrame(bbox_annotations_data)

        # Deserialize bbox schema
        schema_class_path = config.get('bbox_schema_class')
        if not schema_class_path:
            raise ValueError("Missing 'bbox_schema_class' in config")
        
        module_name, class_name = schema_class_path.rsplit(".", 1)
        schema_cls = getattr(importlib.import_module(module_name), class_name)
        
        bbox_schema_data = config.get('bbox_schema')
        if not bbox_schema_data:
            raise ValueError("Missing 'bbox_schema' in config")
        bbox_schema = schema_cls.from_dict(bbox_schema_data)

        return cls(
            **core_kwargs,
            bbox_schema=bbox_schema,
            bbox_annotations=bbox_annotations,
            **kwargs
        )
