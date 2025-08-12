"""
datasets/bbox_dataset.py
"""

from typing import Sequence, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch

from .bbox_schema import BBoxSchema
from .base_dataset import (
    BaseImageDataset,
    TransformType, 
    validate_compose_transform
)

"""

"""
class BBoxCropImageDataset(BaseImageDataset):
    def __init__(
        self,
        file_index: pd.DataFrame,
        bbox_annotations: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        object_metadata: Optional[Sequence[pd.DataFrame]] = None,
        post_crop_transforms: Optional[TransformType] = None,
        bbox_schema: BBoxSchema = BBoxSchema(),
        **kwargs
    ):
        """
        Initializes the BBoxCropImageDataset.

        :param file_index: DataFrame containing file paths to the raw images. 
            Reference BaseImageDataset for details.
        :param bbox_annotations: DataFrame containing bounding box annotations.
            Each row should correspond to a crop defined by the bounding box
            coordinates (xmin, ymin, xmax, ymax) and optionally center, 
            rotation center (rcx, rcy), and angle of rotation. 
            Must match the schema defined by bbox_schema. 
            Must match the length of file_index.
        :param metadata: Optional DataFrame containing metadata associated 
            with each crop. 
            Must match length of file_index.
        :param object_metadata: Optional list of DataFrames containing metadata
            for objects within the crops. Each DataFrame should correspond to
            a crop in the metadata DataFrame.
        :param post_crop_transforms: Optional transformation to apply to crops
            after they are extracted. 
        :param kwargs: Additional keyword arguments passed to BaseImageDataset.
        :param bbox_schema: BBoxSchema object defining the schema for bounding
            box columns. If not provided, defaults to BBoxSchema().
        """
        
        if isinstance(bbox_schema, BBoxSchema):
            self._schema = bbox_schema            
        else:
            raise TypeError(
                "Expected bbox_schema to be of type BBoxSchema, "
                f"got {type(bbox_schema)}"
            )
        
        if not isinstance(bbox_annotations, pd.DataFrame):
            raise TypeError(
                f"Expected metadata to be a DataFrame, got {type(bbox_annotations)}")
        # bbox_annotations = bbox_annotations.copy()
        # bbox_annotations.bbox(self._schema).ensure_columns()
        # self._bbox = bbox_annotations.bbox(self._schema)
    
        # Register the bbox schema, and
        # ensure the necessary columns defining bounding boxes are present
        bbox = bbox_annotations.copy().bbox(self._schema)
        self._bbox_df = bbox.ensure_columns()
        # handle to accessor
        self._bbox = bbox

        # the BaseImageDataset contructor handles the internal
        # attribute initialization binding file_index, metadata, and object_metadata
        super().__init__(
            file_index=file_index,
            metadata=metadata,
            object_metadata=object_metadata,
            **kwargs
        )

        self.post_crop_transforms = post_crop_transforms       

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def _get_raw_item(
        self, 
        idx: int,
        apply_post_crop_transforms: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overridden parent class method. 
        Takes the crops from the raw images and return as
        tuple of numpy arrays.
        """

        # Raw full images from parent as numpy arrays
        raw_input, raw_target = super()._get_raw_item(idx)

        # Get bounding box coordinates and rotation
        xmin, ymin, xmax, ymax = self._bbox.coords(i=idx)
        rcx, rcy = self._bbox.rot_centers(i=idx)
        angle = self._bbox.angle_of(i=idx)

        cropped_input = self._crop_image(
            image=raw_input,
            bbox=(xmin, ymin, xmax, ymax),
            rcx=rcx,
            rcy=rcy,
            angle=angle
        )
        cropped_target = self._crop_image(
            image=raw_target,
            bbox=(xmin, ymin, xmax, ymax),
            rcx=rcx,
            rcy=rcy,
            angle=angle
        )

        if apply_post_crop_transforms:
            return self._apply_post_crop_transform(
                cropped_input, cropped_target
            )
        else:
            return cropped_input, cropped_target

    """
    Post crop transformation can be configured following
    construction of the dataset and will be dynamically applied.
    """
    @property
    def post_crop_transforms(self) -> Optional[TransformType]:
        return self._post_crop_transforms
    @post_crop_transforms.setter
    def post_crop_transforms(self, value: Optional[TransformType] = None):
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=True)
        self._post_crop_transforms = value

    def _apply_post_crop_transform(
        self,
        input_image: np.ndarray,
        target_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.post_crop_transforms is not None:
            result = self.post_crop_transforms(image=input_image, target=target_image)
            return result['image'], result['target']
        return input_image, target_image
    
    """
    These are immutable following construction. Still we allow access
    to the schema and DataFrame containing bounding box annotations.
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
    @property
    def bbox_state(self) -> Tuple[int, int, int, int, float, float, float]:
        """
        Returns the current state of the bounding box annotations as a tuple.
        This includes xmin, ymin, xmax, ymax, rcx, rcy, and angle.
        """

        last_idx = self.index_state.last
        if last_idx is None:
            raise ValueError("Index state is not set. Cannot retrieve bounding box state.")

        xmin, ymin, xmax, ymax = self._bbox.coords(last_idx)
        rcx, rcy = self._bbox.rot_centers(last_idx)
        angle = self._bbox.angle_of(last_idx)

        return (
            xmin, ymin, xmax, ymax, 
            rcx, rcy, angle
        )

    @staticmethod
    def _crop_image(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        rcx: int,
        rcy: int,
        angle: float,
        min_angle: float = 1e-3
    ) -> np.ndarray:
        """
        Internal helper function for cropping image according to the 
        bounding box and optionally applies rotation.

        :param image: The input image to crop.
        :param bbox: The bounding box coordinates (xmin, ymin, xmax, ymax).
        :param rcx: Rotation center x coordinate.
        :param rcy: Rotation center y coordinate.
        :param angle: The angle to rotate the cropped image.
        :param min_angle: The minimum angle threshold for rotation. Below this
            threshold = approximate as having no rotation.
        :return: The cropped (and possibly rotated) image.
        """
        # return fast if no rotation
        xmin, ymin, xmax, ymax = bbox
        if angle == 0.0 or abs(angle) < min_angle or \
            rcx is None or rcy is None:
            return image[:, ymin:ymax, xmin:xmax]

        # Handle both 3D (C, H, W) and 4D (C, H, W, K) cases
        original_shape = image.shape
        
        if image.ndim == 3:
            # Standard case: (C, H, W) -> (H, W, C)
            image_for_cv = np.transpose(image, (1, 2, 0))
        elif image.ndim == 4:
            # 4D case: (C, H, W, K) -> reshape to (H, W, C*K) for cv2
            C, H, W, K = image.shape
            image_for_cv = image.transpose(1, 2, 0, 3).reshape(H, W, C * K)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}. "
                           f"Expected 3D (C, H, W) or 4D (C, H, W, K).")

        # Rotate image around (rcx, rcy)
        M = cv2.getRotationMatrix2D(
            center=(rcx, rcy), 
            angle=angle, 
            scale=1.0
        )
        rotated_cv = cv2.warpAffine(
            image_for_cv,
            M,
            (image_for_cv.shape[1], image_for_cv.shape[0]), # W, H
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Handle single channel case after rotation
        if rotated_cv.ndim == 2:
            rotated_cv = rotated_cv[:, :, np.newaxis]

        # Convert back to original format
        if original_shape.__len__() == 3:
            # Convert back to (C, H, W) format
            rotated_image = np.transpose(rotated_cv, (2, 0, 1))
        else:  # original_shape.__len__() == 4
            # Convert back to (C, H, W, K) format
            C, H, W, K = original_shape
            rotated_H, rotated_W = rotated_cv.shape[:2]
            rotated_image = rotated_cv.reshape(rotated_H, rotated_W, C, K).transpose(2, 0, 1, 3)

        # Extract the same bounding box region from the rotated image
        return rotated_image[:, ymin:ymax, xmin:xmax]
    
    def _serialize_config(self) -> Dict[str, Any]:
        """
        Overridden parent class method to additionally serialize
        the bounding box schema and annotations DataFrame.

        :return: Dictionary containing the serialized configuration.
        """

        config = super()._serialize_config()
        config['bbox_schema'] = self.bbox_schema.to_dict()
        config['bbox_schema_class'] = (
            f"{self.bbox_schema.__class__.__module__}."
            f"{self.bbox_schema.__class__.__name__}"
        )
        config['bbox_annotations'] = self.bbox_df.to_dict('records')

        return config
    
    @classmethod
    def _deserialize_config(
        cls,
        config: Dict[str, Any],
        transform: Optional[TransformType] = None,
        input_only_transform: Optional[TransformType] = None,
        target_only_transform: Optional[TransformType] = None,
        post_crop_transforms: Optional[TransformType] = None
    ) -> 'BBoxCropImageDataset':
        
        from pathlib import Path
        import importlib
        
        # Reuse parent's core deserialization logic
        core_kwargs = cls._deserialize_core_config(config)

        # Handle bbox-specific deserialization
        # Reconstruct bbox_annotations DataFrame
        bbox_annotations_data = config.get('bbox_annotations', None)
        if bbox_annotations_data is None:
            raise ValueError(
                "Expected 'bbox_annotations' in config, "
                "but it is missing or empty."
            )
        bbox_annotations = pd.DataFrame(bbox_annotations_data)

        # Reconstruct bbox_schema
        bbox_scheme_class = config.get('bbox_schema_class')
        if not bbox_scheme_class:
            raise ValueError(
                "Expected 'bbox_schema_class' in config, "
                "but it is missing or empty."
            )
        module_name, class_name = bbox_scheme_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        schema_cls = getattr(module, class_name)
        bbox_schema_data = config.get('bbox_schema', None)
        if bbox_schema_data is None:
            raise ValueError(
                "Expected 'bbox_schema' in config, "
                "but it is missing or empty."
            )
        bbox_schema = schema_cls.from_dict(bbox_schema_data)

        return cls(
            **core_kwargs,
            bbox_schema=bbox_schema,
            bbox_annotations=bbox_annotations,
            post_crop_transforms=post_crop_transforms,
            transform=transform,
            input_only_transform=input_only_transform,
            target_only_transform=target_only_transform
        )