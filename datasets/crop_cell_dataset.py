from pathlib import Path
from typing import Optional, List, Sequence, Union, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

from .base_dataset import BaseImageDataset, TransformType, validate_compose_transform
from .bbox_dataset import BBoxCropImageDataset
from .bbox_schema import BBoxSchema
from .crop_generator import ObjectCenteredCropGenerator

class CropCellImageDataset(BBoxCropImageDataset):

    def __init__(
        self,
        patch_size: int,
        file_index: pd.DataFrame,
        object_metadata: List[pd.DataFrame],
        object_coord_x_field: str,
        object_coord_y_field: str,
        metadata: Optional[pd.DataFrame] = None,
        fov: Optional[Tuple[int, int]] = None,
        bbox_schema: BBoxSchema = BBoxSchema(),
        **kwargs
    ):
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if not isinstance(object_coord_x_field, str):
            raise TypeError("object_coord_x_field must be a string.")
        if not isinstance(object_coord_y_field, str):
            raise TypeError("object_coord_y_field must be a string.")
        
        self._patch_size = patch_size
        self._object_coord_x_field = object_coord_x_field
        self._object_coord_y_field = object_coord_y_field

        crop_generator = ObjectCenteredCropGenerator(
            crop_size=patch_size,
            object_coord_x_field=object_coord_x_field,
            object_coord_y_field=object_coord_y_field,
            fov=fov,
            bbox_schema=bbox_schema
        )
        
        # Generate crops
        bboxes_df_list, bboxes_obj_metadata_list = crop_generator.generate_crops(
            file_index, object_metadata
        )

        # For ImageDataset to work, the view/site level dataframes
        # need to be expanded to match the number of bounding boxes.
        # Specifically, the file_index and metadata DataFrames are expanded
        # below:
        expanded_file_index, _ = expand_df_match_bbox(
            file_index, bboxes_df_list
        )
        expanded_metadata, _ = expand_df_match_bbox(
            metadata, bboxes_df_list
        ) if metadata is not None else None

        all_bboxes_df = pd.concat(
            bboxes_df_list, ignore_index=True).reset_index(drop=True)
        
        # Sort the DataFrames to ensure consistent order
        order, expanded_file_index, all_bboxes_df = deterministic_dual_sort(
            expanded_file_index, all_bboxes_df,
            id_cols=None, # uses all columns of the bbox_df
            case_insensitive=True
        )

        # If metadata was provided, sort it as well
        if expanded_metadata is not None:
            expanded_metadata = expanded_metadata.loc[
                order].reset_index(drop=True)

        # Reorder the object metadata list in place to match the sorted order
        reorder_list_inplace(bboxes_obj_metadata_list, order)
        
        super().__init__(
            file_index=expanded_file_index,
            bbox_annotations=all_bboxes_df,
            metadata=expanded_metadata,
            object_metadata=bboxes_obj_metadata_list,
            bbox_schema=bbox_schema,
            **kwargs
        )
    
    @classmethod
    def from_dataset(
        cls,
        dataset: BaseImageDataset,
        patch_size: int,
        object_coord_x_field: str,
        object_coord_y_field: str,
        fov: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> "CropCellImageDataset":
        """
        Create a CropCellDataset from an existing BaseImageDataset.

        :param dataset: The BaseImageDataset to crop from.
        :param patch_size: Size of square crops.
        :param object_coord_x_field: Column name for x-coordinate in object metadata.
        :param object_coord_y_field: Column name for y-coordinate in object metadata.
        :param fov: Optional FOV shape override (height, width).
        :return: A new CropCellDataset instance.
        """
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("dataset must be an instance of BaseImageDataset.")

        return cls(
            patch_size=patch_size,
            file_index=dataset.file_index,
            metadata=dataset.metadata,
            object_metadata=dataset.object_metadata,
            object_coord_x_field=object_coord_x_field,
            object_coord_y_field=object_coord_y_field,
            fov=fov,
            pil_image_mode=dataset.pil_image_mode,
            input_channel_keys=dataset.input_channel_keys,
            target_channel_keys=dataset.target_channel_keys,
            **kwargs
        )

"""
Helper methods for wrangling the file_index, metadata and object_metadata
DataFrames for a image crop dataset.
"""
def expand_df_match_bbox(
    df: pd.DataFrame,
    bboxes_df_list: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Expand the DataFrame to match the number of bounding boxes per image.

    :param df: DataFrame with metadata.
    :param bboxes_df_list: Correponding List of DataFrames describing 
        bounding boxes for each view/site. len(bboxes_df_list) should 
        match len(df).
    """

    box_counts = [len(bboxes) for bboxes in bboxes_df_list]

    if len(box_counts) != len(df):
        raise ValueError(
            "Length of bboxes_df_list must match the length of df."
        )
    
    mapping = np.repeat(np.arange(len(df)), box_counts)

    expanded_df = df.iloc[mapping].reset_index(drop=True)

    return expanded_df, mapping

def deterministic_dual_sort(
        file_index: pd.DataFrame,
        bbox_df: pd.DataFrame,
        id_cols=None,
        case_insensitive=False,
        na_position="last"
    ):
    """
    Perform a stable lexicographic sort on two DataFrames:
    - file_index: DataFrame containing file paths or identifiers.
    - bbox_df: DataFrame containing bounding box annotations.
    This ensures the order of the dataset constructed from the file_index
        and bbox_df is consistent across runs.
    """
    assert len(file_index) == len(bbox_df), "DataFrames must be row-aligned and same length."

    # Normalize pathlib columns to strings
    norm = file_index.map(
        lambda p: p.as_posix().lower() if case_insensitive else p.as_posix())

    # Select numeric columns to sort by (default: all)
    if id_cols is None:
        id_cols = list(bbox_df.columns)

    # Build a temporary key frame: all path cols first, then numeric cols
    key_df = pd.concat([norm, bbox_df[id_cols]], axis=1)

    # Stable lexicographic sort across all key columns
    order = key_df.sort_values(list(key_df.columns),
                               kind="mergesort",    # stable
                               na_position=na_position).index

    # Apply the same order to both original DataFrames
    file_index_sorted   = file_index.loc[order].reset_index(drop=True)
    bbox_df_sorted = bbox_df.loc[order].reset_index(drop=True)

    return order, file_index_sorted, bbox_df_sorted

def reorder_list_inplace(lst, order_pos):
    """
    Memory efficiently reorder a list in place
    """
    # Build inverse permutation: inv[orig_pos] = new_pos
    n = len(lst)
    inv = [None] * n
    for new_pos, orig_pos in enumerate(order_pos):
        inv[orig_pos] = new_pos

    # Cycle decomposition with swap-based permute (mutates inv as we go)
    for i in range(n):
        while inv[i] != i:
            j = inv[i]
            lst[i], lst[j] = lst[j], lst[i]
            inv[i], inv[j] = inv[j], inv[i]