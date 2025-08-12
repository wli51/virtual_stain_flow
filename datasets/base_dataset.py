"""
base_dataset.py

This file contains the `BaseImageDataset` class, meant to serve as the foundation
infranstructure for all image datasets in the `datasets` module, defining
the basic attribute structure, signature, properties to shared across all 
image datasets.
"""

from typing import Dict, Sequence, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations import Compose, ImageOnlyTransform

from .utils import _to_hwc, _to_chw
from .dataset_view import DatasetView, IndexState, FileState

# temporary alias here for type hinting
# will be moved/centralized to the transform module when the refactor is complete
TransformType = Union[ImageOnlyTransform, Compose]
def validate_compose_transform(
    obj: TransformType,
    apply_to_target: bool = True,
) -> Compose:
    """
    Validates and returns a Compose transform.
    Likewise temporary place for this function, to be cenralized in transform module
    when the refactor is complete.
    
    :param obj: The transform object to validate.
    :param apply_to_target: Whether the transform should be applied to target images.
    :return: Validated Compose transform.
    :raises TypeError: If the transform is not valid.
    """
    add_targets = {'target': 'image'} if apply_to_target else {}

    if isinstance(obj, ImageOnlyTransform):
        return Compose([obj], additional_targets=add_targets)

    elif isinstance(obj, Sequence):
        if not all(isinstance(t, ImageOnlyTransform) for t in obj):
            raise TypeError("All items must be ImageOnlyTransform instances.")
        return Compose(list(obj), additional_targets=add_targets)

    elif isinstance(obj, Compose):
        if apply_to_target and 'target' not in getattr(obj, 'additional_targets', {}):
            raise ValueError(
                "apply_to_target=True requires 'target' in Compose.additional_targets."
            )
        return obj
    else:
        raise TypeError(
            f"Expected Compose, ImageOnlyTransform, or Sequence[ImageOnlyTransform], got {type(obj)}."
        )

class BaseImageDataset(Dataset):
    
    def __init__(
        self,
        file_index: pd.DataFrame,
        pil_image_mode: str = "I;16",
        metadata: Optional[pd.DataFrame] = None,
        object_metadata: Optional[Sequence[pd.DataFrame]] = None,
        input_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        target_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        transform: Optional[TransformType] = None,
        input_only_transform: Optional[TransformType] = None,
        target_only_transform: Optional[TransformType] = None,
        cache_capacity: Optional[int] = None,
    ):
        """
        Initializes the BaseImageDataset.

        :param file_index: DataFrame containing exclusively file paths as pathlikes
        :param pil_image_mode: Mode for PIL images, default is "I;16".
        :param metadata: Optional DataFrame with additional metadata.
        :param object_metadata: Optional list of DataFrames with object-level metadata.
        :param input_channel_keys: Keys for input channels in the file index.
        :param target_channel_keys: Keys for target channels in the file index.
        :param transform: Transformations to apply to both input and target images.
        :param input_only_transform: Transformations to apply to input images.
        :param target_only_transform: Transformations to apply to target images.
        :param cache_capacity: Optional capacity for caching loaded images. 
            When set to None, default caching behavior of caching at most
            `file_index.shape[0]` images is used. When set to -1, unbounded
            caching without eviction is used. When set to a positive integer,
            the cache will hold at most that many images, evicting the least recently
            used images when the cache is full (LRU cache). Other values are
            invalid.                 
        """

        self.view = DatasetView(
            file_index=file_index, 
            pil_image_mode=pil_image_mode
        )
        self.index_state = IndexState()

        self.file_state = FileState(
            view=self.view, 
            cache_capacity=cache_capacity
        )
        
        self.metadata: pd.DataFrame = metadata
        self.object_metadata: Sequence[pd.DataFrame] = object_metadata
        
        self.input_channel_keys = input_channel_keys
        self.target_channel_keys = target_channel_keys

        self.transform = transform
        self.input_only_transform = input_only_transform
        self.target_only_transform = target_only_transform

    """
    Retrieval backend methods
    """

    def _apply_transform(
        self,
        input_image: np.ndarray,
        target_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:        
        """
        Helper method to apply the dataset's configured image transformation
        to an input and target image stack pair.
        
        :param input_image: Input image (C, H, W) as a NumPy array.
        :param target_image: Target image (C, H, W) as a NumPy array.
        :return: Tuple of transformed input and target images.
        """

        inp_ref, tgt_ref = input_image, target_image

        # Albumentations expects HWC
        inp_hwc = _to_hwc(input_image)
        tgt_hwc = _to_hwc(target_image)

        # First apply shared transforms
        if self.transform is not None:
            t = self.transform(image=inp_hwc, target=tgt_hwc)
            inp_hwc = t['image']
            tgt_hwc = t['target']
        
        # Then apply input-only and target-only transforms
        if self.input_only_transform is not None:
            inp_hwc = self.input_only_transform(image=inp_hwc)['image']

        if self.target_only_transform is not None:
            tgt_hwc = self.target_only_transform(image=tgt_hwc)['image']

        # Convert back to CHW (match original shapes)
        inp_chw = _to_chw(inp_hwc, ref=inp_ref)
        tgt_chw = _to_chw(tgt_hwc, ref=tgt_ref)
        
        return inp_chw, tgt_chw

    def _get_raw_item(
            self, 
            idx: int
        ) -> Tuple[np.ndarray, np.ndarray]:

        self.index_state.update(idx)

        # load files lazily given current channel config
        self.file_state.update(
            idx, 
            input_keys=self.input_channel_keys, 
            target_keys=self.target_channel_keys
        )

        return self._apply_transform(
            self.file_state.input_image_raw,
            self.file_state.target_image_raw
        )    

    """
    Dataset `__len__` and `__getitem__` methods are overridden
    """
    def __len__(self) -> int:
        """
        Overridden Dataset `__len__` method so class works with torch DataLoader.
        """
        return len(self.view)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overridden Dataset `__getitem__` method so class works with torch DataLoader.
        """        
        input, target = self._get_raw_item(idx)

        return torch.from_numpy(input).float(), torch.from_numpy(target).float()    
    
    """
    Properties for dataset attributes
    """

    @property
    def pil_image_mode(self) -> str:
        """
        Returns the PIL image mode.
        """
        return self.view.pil_image_mode

    @property
    def file_index(self) -> pd.DataFrame:
        """
        Returns the file index DataFrame.
        The `file_index` attribute of the dataset class is expected to be
            immutable after class initialization, hence no setter is provided.
        """
        return self.view.file_index

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Returns the metadata DataFrame.
        """
        return self._metadata
    @metadata.setter
    def metadata(self, value: Optional[pd.DataFrame]=None):
        """
        Sets the metadata DataFrame.
        Metadata is expected to be a pandas DataFrame with the same
            length as the file index.
        """
        if value is None:
            value = pd.DataFrame(index=range(len(self.view)))
        elif not isinstance(value, pd.DataFrame):
            raise ValueError("Expected metadata to be a pandas DataFrame, "
                             f"got {type(value)} instead.")
        elif len(value) != len(self.view):
            raise ValueError(f"Length of metadata must match the length of "
                             f"file_index {len(self.view)}. "
                             f"Got {len(value)} instead.")
        self._metadata = value

    @property
    def object_metadata(self) -> Sequence[pd.DataFrame]:
        """
        Returns the object metadata DataFrames.
        """
        return self._object_metadata
    @object_metadata.setter
    def object_metadata(self, value: Optional[Sequence[pd.DataFrame]]=None):
        """
        Sets the object metadata DataFrames.
        Object metadata is expected to be a sequence of pandas DataFrames. 
            The sequence length must match the length of the file index.
            Each element datafarme can be of arbitrary length. 
        """
        if value is None:
            value = [pd.DataFrame() for _ in range(len(self.view))]
        elif not isinstance(value, Sequence):
            raise ValueError("Expected object_metadata to be a sequence of "
                             "pandas DataFrames, "
                             f"got {type(value)} instead.")
        elif len(value) != len(self.view):
            raise ValueError(f"Length of object_metadata must match the length "
                             "of "
                             f"file_index {len(self.view)}. "
                             f"Got {len(value)} instead.")
        self._object_metadata = value

    def _validate_channel_keys(
        self,
        channel_keys: Optional[Union[str, Sequence[str]]],
    ):
        """
        Validates the channel keys against the file index columns.
        
        :param channel_keys: Keys for input or target channels.
        :raises ValueError: If channel_keys is invalid.
        """
        if channel_keys is None:
            return []
        elif isinstance(channel_keys, str):
            channel_keys = [channel_keys]
        elif not isinstance(channel_keys, Sequence):
            raise ValueError("Expected channel_keys to be a string or a "
                             "sequence of strings, "
                             f"got {type(channel_keys)} instead.")
        
        for key in channel_keys:
            if key not in self.view.file_index.columns:
                raise ValueError(f"Channel key '{key}' not found in "
                                 "file_index columns.")            
        return channel_keys    

    @property
    def input_channel_keys(self) -> Optional[Union[str, Sequence[str]]]:
        """
        Returns the input channel keys.
        """
        return self._input_channel_keys
    @input_channel_keys.setter
    def input_channel_keys(self, value: Optional[Union[str, Sequence[str]]]=None):
        """
        Sets the input channel keys.
        """
        value = self._validate_channel_keys(value)        
        self._input_channel_keys = value

    @property
    def target_channel_keys(self) -> Optional[Union[str, Sequence[str]]]:
        """
        Returns the target channel keys.
        """
        return self._target_channel_keys
    @target_channel_keys.setter
    def target_channel_keys(self, value: Optional[Union[str, Sequence[str]]]=None):
        """
        Sets the target channel keys.
        """
        value = self._validate_channel_keys(value)        
        self._target_channel_keys = value

    @property
    def transform(self) -> Optional[TransformType]:
        """
        Returns the transform applied to both input and target images.
        """
        return self._transform
    @transform.setter
    def transform(self, value: Optional[TransformType]=None):
        """
        Sets the transform to be applied to both input and target images.
        """
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=True)
        self._transform = value
    
    @property
    def input_only_transform(self) -> Optional[TransformType]:
        """
        Returns the input-only transform.
        """
        return self._input_only_transform
    @input_only_transform.setter
    def input_only_transform(self, value: Optional[TransformType]=None):
        """
        Sets the input-only transform.
        """
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=False)
        self._input_only_transform = value

    @property
    def target_only_transform(self) -> Optional[TransformType]:
        """
        Returns the target-only transform.
        """
        return self._target_only_transform
    @target_only_transform.setter
    def target_only_transform(self, value: Optional[TransformType]=None):
        """
        Sets the target-only transform.
        """
        if value is not None:
            value = validate_compose_transform(value, apply_to_target=False)
        self._target_only_transform = value

    @property
    def last_loaded_index(self) -> Optional[int]:
        """
        Returns the last loaded index from the dataset.
        """
        return self.index_state.last
    
    @property
    def metadata_state(self) -> pd.Series:
        """
        Returns the current metadata state as a pandas Series.
        The Series is indexed by the file index.
        """
        if self.index_state.last is None:
            raise RuntimeError(
                "No index has been loaded yet. "
                "Call __getitem__ or _get_raw_item first."
            )
        
        return self._metadata.iloc[self.index_state.last]    
    @property
    def object_metadata_state(self) -> pd.DataFrame:
        """
        Returns the current object metadata state as a pandas DataFrame.
        The DataFrame is indexed by the file index.
        """
        if self.index_state.last is None:
            raise RuntimeError(
                "No index has been loaded yet. "
                "Call __getitem__ or _get_raw_item first."
            )        
        return self._object_metadata[self.index_state.last]