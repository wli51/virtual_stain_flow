"""
base_dataset.py

This file contains the `BaseImageDataset` class, meant to serve as the foundation
infranstructure for all image datasets in the `datasets` module, defining
the basic attribute structure, signature, properties to shared across all 
image datasets.
"""

from typing import Dict, Sequence, Optional, Tuple, Union, Any
import json
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations import Compose, ImageOnlyTransform, BasicTransform

from .utils import _to_hwc, _to_chw
from .dataset_view import DatasetView, IndexState, FileState
from ..transforms import TransformType, validate_compose_transform

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

    def to_json_config(self, filepath: Union[str, Path]) -> None:
        """
        Exposed method for serializing the dataset as a JSON file, 
        facilitating reproducibility by saving all information needed 
        to reconstruct the dataset. At the moment transforms are not
        serializable and hence are ignored. Future development may
        address this limitation.

        :param filepath: Path where to save the JSON configuration file.
        """
        config = self._serialize_config()
        
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _serialize_config(self) -> Dict[str, Any]:
        """
        Internal method to serialize the dataset configuration to a dictionary.
        
        :return: Dictionary containing the serialized configuration.
        """
        # Convert file_index to records format for JSON serialization
        # Convert Path objects to strings for JSON compatibility
        file_index_for_json = self.file_index.copy()
        for col in file_index_for_json.columns:
            file_index_for_json[col] = file_index_for_json[col].apply(
                lambda x: str(x) if isinstance(x, (Path, PurePath)) else x
            )
        
        file_index_records = file_index_for_json.to_dict('records')
        file_index_columns = list(self.file_index.columns)
        
        # Serialize metadata DataFrame
        metadata_records = None
        if self.metadata is not None and not self.metadata.empty:
            metadata_records = self.metadata.to_dict('records')
        
        # Serialize object_metadata list of DataFrames
        object_metadata_records = []
        if self.object_metadata is not None:
            for df in self.object_metadata:
                if df is not None and not df.empty:
                    object_metadata_records.append(df.to_dict('records'))
                else:
                    object_metadata_records.append([])
        
        config = {
            'file_index': {
                'records': file_index_records,
                'columns': file_index_columns
            },
            'pil_image_mode': self.pil_image_mode,
            'metadata': metadata_records,
            'object_metadata': object_metadata_records,
            'input_channel_keys': self.input_channel_keys,
            'target_channel_keys': self.target_channel_keys,
            'cache_capacity': self.file_state.cache_capacity,
            'dataset_length': len(self)
        }
        
        return config

    @classmethod
    def from_json_config(
        cls, 
        filepath: Union[str, Path],
        transform: Optional[TransformType] = None,
        input_only_transform: Optional[TransformType] = None,
        target_only_transform: Optional[TransformType] = None,
    ) -> 'BaseImageDataset':
        """
        Create a BaseImageDataset instance from a JSON configuration file.
        Wraps around _deserialize_config with an extra file reading step.

        :param filepath: Path to the JSON configuration file.
        :param transform: Optional transform to apply to both input and target images.
        :param input_only_transform: Optional transform to apply only to input images.
        :param target_only_transform: Optional transform to apply only to target images.
        :return: BaseImageDataset instance.
        """
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls._deserialize_config(
            config,
            transform=transform,
            input_only_transform=input_only_transform,
            target_only_transform=target_only_transform
        )

    @classmethod
    def _deserialize_core_config(
        cls,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and deserialize core dataset configuration 
            components and returns a dictionary of kwargs ready
            to be passed to the BaseImageDataset constructor.

        Subclass deserialization methods may call this method
            to conveniently extract the common configuration.
        """
        # Reconstruct file_index DataFrame
        file_index_data = config.get('file_index', None)
        if file_index_data is None:
            raise ValueError(
                "Expected 'file_index' in config, "
                "but found none or empty."
            )
        
        file_index = pd.DataFrame(file_index_data['records'])
        # Convert string paths back to Path objects
        for col in file_index.columns:
            file_index[col] = file_index[col].apply(
                lambda x: Path(x) if isinstance(x, str) else x
            )

        pil_image_mode = config.get('pil_image_mode', 'I;16')
        
        # Reconstruct metadata DataFrame
        metadata_data = config.get('metadata', None)
        metadata = None
        if metadata_data is not None:
            metadata = pd.DataFrame(metadata_data)
        
        # Reconstruct object_metadata list of DataFrames
        object_metadata_data = config.get('object_metadata', None)
        object_metadata = None
        if object_metadata_data is not None:
            object_metadata = []
            for records in config['object_metadata']:
                if records:
                    object_metadata.append(pd.DataFrame(records))
                else:
                    object_metadata.append(pd.DataFrame())

        
        input_channel_keys=config.get('input_channel_keys', None)
        target_channel_keys=config.get('target_channel_keys', None)
        cache_capacity=config.get('cache_capacity', None)
                    
        return {
            'file_index': file_index,
            'pil_image_mode': pil_image_mode,
            'metadata': metadata,
            'object_metadata': object_metadata,
            'input_channel_keys': input_channel_keys,
            'target_channel_keys': target_channel_keys,
            'cache_capacity': cache_capacity
        }

    @classmethod
    def _deserialize_config(
        cls,
        config: Dict[str, Any],
        transform: Optional[TransformType] = None,
        input_only_transform: Optional[TransformType] = None,
        target_only_transform: Optional[TransformType] = None,
    ) -> 'BaseImageDataset':
        """
        Internal method to deserialize a configuration dictionary.
        Because this is the base call, no further deserialization
            is needed beyond calling `_deserialize_core_config`.
        Subclass should override this method and use 
            `_deserialize_core_config` as is to extract the common 
            configuration.
        
        :param config: Configuration dictionary.
        :param transform: Optional transform to apply to both input and target images.
        :param input_only_transform: Optional transform to apply only to input images.
        :param target_only_transform: Optional transform to apply only to target images.
        :return: BaseImageDataset instance.
        """        

        # when overriding this method, subclasses should still call
        core_ds_kwargs = cls._deserialize_core_config(config)

        # do some more deserialization
        #core_ds_kwargs['additional_complexity'] = ...

        return cls(
            **core_ds_kwargs,
            transform=transform,
            input_only_transform=input_only_transform,
            target_only_transform=target_only_transform,
        )