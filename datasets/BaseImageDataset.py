import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..transforms import (
    TransformType, 
    TransformTypeRuntime,
    transform_type_error_text
    )

class BaseImageDataset(Dataset):
    """
    Base PyTorch Image Dataset class for dynamically loading and transforming images.
    Intended as a bare-minimum dataset class requiring a formatted file index DataFrame
        containing file paths of images organized across views and channels. Has placeholder
        attributes for view-wise and object-wise (within view) metadata that can be 
        populated manually after initialization or by subclasses. 
    This class also is meant to centralize shared image dataset functionalites, 
        for now these include: 
        - configuration of input/target transformation
        - dataset export/loading (for reproducibility).
        - access of last retrieved image filepaths (`input_names` and `target_names`) 
            and metadata (`current_metadata` and `current_object_metadata`).
    Subclasses may wrap this base class with more sophisticated file indexing logic for 
        more convenient data loading.

    TODO: currently all of the configurable attributes have a python setter method
        as well as a wrapper method that merely calls the setter (`set_<attribute_name>()`)
        for more intuitive use.
        We might want to eliminate one of the two. 
    """

    def __init__(
            self, 
            file_index: pd.DataFrame, 
            pil_image_mode: str = "I;16",
            metadata: Optional[pd.DataFrame] = None,
            object_metadata: Optional[List[pd.DataFrame]] = None,
            input_channel_keys: Optional[Union[str, List[str]]] = None,
            target_channel_keys: Optional[Union[str, List[str]]] = None,
            input_transform: TransformType = None,
            target_transform: TransformType = None
        ):
        """
        Initializes the Dataset class with a file index, image mode, channels, and optional metadata.
        This is intended as a very low-level base dataset class for images requriing non-trivial manual
            compilation of the file_index dataframe, which is a (n_views, n_channels) DataFrame containing
            file paths of tiff images organized across views and channels.
        Channel keys for input and target images can be specified after initialization with 
            methods `set_input_channel_keys` and `set_target_channel_keys`.
        
        :param file_index: DataFrame containing file paths for input and target images.
            Note that this class does not validate the file paths or completeness of the index.
        :param pil_image_mode: Mode for PIL image conversion (default is "I;16" for 16-bit images).
        :param metadata: Optional DataFrame containing metadata for the dataset.
        :param object_metadata: Optional list of DataFrames containing object-level metadata for each view.
        :param input_channel_keys: List of keys for input image channels.
        :param target_channel_keys: List of keys for target image channels.
        :param input_transform: Optional transformation to apply to input images.
        :param target_transform: Optional transformation to apply to target images.
        """

        if not isinstance(file_index, pd.DataFrame):
            raise TypeError("file_index must be a pandas DataFrame.")
        if file_index.empty:
            raise ValueError("file_index cannot be empty.")
        self.file_index: pd.DataFrame = file_index
        if not isinstance(pil_image_mode, str):
            raise TypeError("pil_image_mode must be a string.")
        self._pil_image_mode: str = pil_image_mode

        self._metadata: pd.DataFrame = None
        self.set_metadata(metadata)
        # TODO: not sure if the variable name `_object_metadata` is final
        # this may be useful for subclasses that require object-level metadata
        # for operations like cropping object containing patches.
        self._object_metadata: List[pd.DataFrame] = None
        self.set_object_metadata(object_metadata)

        # Configure input and target channel keys if provided,
        # otherwise initialize as empty lists.
        self.input_channel_keys: List[str] = input_channel_keys
        self.target_channel_keys: List[str] = target_channel_keys

        # Configure input and target transformations if provided,
        # otherwise initialize as None to indicate no transformation.
        self._input_transform = None
        self.input_transform = input_transform
        self._target_transform = None
        self.target_transform = target_transform

        # ------- Internal state to track last accessed index and images --------
        self._last_loaded_idx: Optional[int] = None

        # last loaded single view images are stored as tuple of input, target stack
        # where each stack is a numpy array of shape (n_channels, height, width)
        # the ordering of channels in the stacks is determined by the user-specified
        # `input_channel_keys` and `target_channel_keys`.
        self._last_loaded_images: Optional[Tuple[np.ndarray, np.ndarray]] = None 
        self._last_loaded_input_names: Optional[List[str]] = None
        self._last_loaded_target_names: Optional[List[str]] = None
        self._last_loaded_metadata: Optional[pd.Series] = None
        self._last_loaded_object_metadata: Optional[pd.DataFrame] = None

    def _reset_last_loaded(self):
        """Reset the last loaded images and index."""
        self._last_loaded_idx = None
        self._last_loaded_images = None
        self._last_loaded_input_names = None
        self._last_loaded_target_names = None
        self._last_loaded_metadata = None
        self._last_loaded_object_metadata = None

    def _check_format_keys(
            self, 
            keys: Optional[Union[str, List[str]]] = None
            ):
        """Check if the provided keys are valid."""
        if keys is None:
            return []
        
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, List):
            for k in keys:
                if k not in self.file_index.columns:
                    raise ValueError(f"Channel Key '{k}' not found in file index columns: "
                                    f"{self.file_index.columns.tolist()}")
                
        return keys
    
    """
    Properties and Setters for Input and Target Channel Keys
    """

    @property
    def input_channel_keys(self) -> List[str]:
        """Get the keys for input channels."""
        return self._input_channel_keys
    
    @input_channel_keys.setter
    def input_channel_keys(self, keys: Union[str, List[str]]):
        """
        Set the keys for input channels.
        Validates the keys against the file index columns.
        
        :param keys: Single key or list of keys for input channels.
        """
        keys = self._check_format_keys(keys)
        self._input_channel_keys = keys

        # when the user has re-configured the input/target channel keys,
        # the last loaded images becomes stale so a reset is needed
        self._reset_last_loaded()

    def set_input_channel_keys(self, keys: Union[str, List[str]]):
        """
        Set the keys for input channels.
        Merely wraps the setter for input_channel_keys.
        """
        self.input_channel_keys = keys

    @property
    def target_channel_keys(self) -> List[str]:
        """Get the keys for target channels."""
        return self._target_channel_keys
    
    @target_channel_keys.setter
    def target_channel_keys(self, keys: Union[str, List[str]]):
        """
        Set the keys for target channels.
        Validates the keys against the file index columns.
        
        :param keys: Single key or list of keys for target channels.
        """
        keys = self._check_format_keys(keys)
        self._target_channel_keys = keys

        # when the user has re-configured the input/target channel keys,
        # the last loaded images becomes stale so a reset is needed
        self._reset_last_loaded()

    def set_target_channel_keys(self, keys: Union[str, List[str]]):
        """
        Set the keys for target channels.
        Merely wraps the setter for target_channel_keys.
        """
        self.target_channel_keys = keys
    
    """
    Input and Target Transformations
    """

    def _check_transform(self, transform: TransformType):
        """
        Check if the provided transform is valid.

        :param transform: Transformation to validate.
        :raise TypeError: If transform is not None or an instance of albumentations.ImageOnlyTransform or albumentations.Compose.
        """
        if not isinstance(transform, TransformTypeRuntime):
            raise TypeError(transform_type_error_text + f"Received: {type(transform)}")    
    
    @property
    def input_transform(self) -> TransformType:
        """Get the current input transformation."""
        return self._input_transform
    
    @input_transform.setter
    def input_transform(self, transform: TransformType):
        """
        Set the transformation for inputs.

        :param transform: Transformation to apply to input images.
        """
        self._check_transform(transform)
        self._input_transform = transform
        # note that no `_reset_last_loaded` is necessary here as 
        # the transformations are applied in `__getitem__` only
        # **after** access of last loaded image (stored as raw) or reading in new images.

    def set_input_transform(self, transform: TransformType):
        """
        Set the transformation for input images.
        Merely wraps the setter for input_transform.
        """
        self.input_transform = transform

    @property
    def target_transform(self) -> TransformType:
        """Get the current target transformation."""
        return self._target_transform

    @target_transform.setter
    def target_transform(self, transform: TransformType):
        """
        Set the transformation for targets.

        :param transform: Transformation to apply to target images.
        """
        self._check_transform(transform)
        self._target_transform = transform
        # note that no `_reset_last_loaded` is necessary here as 
        # the transformations are applied in `__getitem__` only
        # **after** access of last loaded image (stored as raw) or reading in new images.

    def set_target_transform(self, transform: TransformType):
        """
        Set the transformation for target images.
        Merely wraps the setter for target_transform.
        """
        self.target_transform = transform

    """
    Dataset Length and Item Retrieval
    """

    def _check_idx(self, idx: int):
        """
        Check if the provided index is valid for the dataset.

        :raise TypeError: If idx is not an integer.
        :raise IndexError: If idx is out of range for the dataset.
        """        
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer.")
        if idx < 0 or idx >= len(self.file_index):
            raise IndexError("Index out of range for the dataset.")
    
    def __len__(self) -> int:
        """Return the number of views in the dataset."""
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve images for the specified index, apply transformations, and return them.

        :param idx: Index of the view to retrieve.
        :return: Tuple of input and target images as numpy arrays.
        """
        if not self.input_channel_keys or not self.target_channel_keys:
            raise ValueError("Input and target channels must be set before accessing data.")
        
        self._check_idx(idx)

        if self._last_loaded_idx is None or self._last_loaded_idx != idx:
            # Load image channels into memory on demand
            input_images, input_names = self._load_images(idx, self.input_channel_keys)
            target_images, target_names = self._load_images(idx, self.target_channel_keys)

            self._last_loaded_images = (input_images, target_images)
            self._last_loaded_input_names = input_names
            self._last_loaded_target_names = target_names
            self._last_loaded_idx = idx
            self._last_loaded_metadata = self._metadata.iloc[idx]
            self._last_loaded_object_metadata = self._object_metadata[idx]

        input_images, target_images = self._last_loaded_images

        # Transform applied outside of the last loaded images cache 
        # to allow for dynamic transformations that gets re-applied
        if self.input_transform:
            input_images = self.input_transform(image=input_images)['image']
        if self.target_transform:
            target_images = self.target_transform(image=target_images)['image']

        return input_images, target_images
    
    """
    Last Loaded Image Metadata Accessors
    """

    @property
    def pil_image_mode(self) -> str:
        """Get the PIL image mode."""
        return self._pil_image_mode
    
    @property
    def channel_keys(self) -> List[str]:
        """Get a list of all available channel keys in the file index."""
        return self.file_index.columns.tolist()

    @property
    def input_names(self) -> List[str]:
        """List of last accessed input filenames."""
        return self._last_loaded_input_names

    @property
    def target_names(self) -> List[str]:
        """List of last accessed target filenames."""
        return self._last_loaded_target_names
    
    @property
    def current_metadata(self) -> pd.Series:
        """Get the metadata for the last accessed index."""
        if self._last_loaded_metadata is None:
            raise ValueError("No metadata loaded. Access an item first.")
        return self._last_loaded_metadata
    
    @property
    def current_object_metadata(self) -> pd.DataFrame:
        """Get the object metadata for the last accessed index."""
        if self._last_loaded_object_metadata is None:
            raise ValueError("No object metadata loaded. Access an item first.")
        return self._last_loaded_object_metadata

    """
    Image Loading and Metadata Management
    """

    def _load_images(self, idx: int, keys: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Load images from disk for the given index and specified channel keys.
        Intended as a backend for `__getitem__` to load images on demand, 
            hence does not perform idx or channel key validation.

        :param idx: Index of the view to retrieve.
        :param keys: List of keys corresponding to the channels to load.
        :return: Tuple of image stack as numpy array and list of filenames.
        """

        raw_images = []
        filenames = []
        view_data = self.file_index.iloc[idx]

        for key in keys:
            if pd.isna(view_data[key]):
                raise ValueError(
                    f"Missing file for the key '{key}' at index '{idx}'. "
                    "Ensure all specified image files/channels are present")
            image_path = view_data[key]
            try:
                with Image.open(image_path) as img:
                    raw_images.append(img.convert(self.pil_image_mode))
                    filenames.append(image_path)
            except FileNotFoundError:
                raise ValueError(f"Image file not found: {image_path}")
        image_stack = np.stack(raw_images, axis=0)

        return image_stack, filenames
    
    @property
    def metadata(self) -> pd.DataFrame:
        """Get the metadata DataFrame."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata: Optional[pd.DataFrame]=None):
        """
        Set the metadata for the dataset.
        Overwrites any existing metadata. 
        If None is provided, sets an empty DataFrame with row count matching file_index.

        :param metadata: DataFrame containing metadata to set. If None, initializes an empty DataFrame.
        """
        if metadata is None:
            self._metadata = pd.DataFrame(index=range(len(self.file_index)))
            return

        if isinstance(metadata, pd.DataFrame):
            if len(metadata) != len(self.file_index):
                raise ValueError("Metadata length must match the number of views in the dataset.")
        else:
            raise TypeError(
                "Expected metadata to be a pandas DataFrame, "
                f"got {type(metadata).__name__} instead."
            )

        self._metadata = metadata
    
    def set_metadata(
            self,
            metadata: Optional[pd.DataFrame]=None,
        ):
        """
        Wrapper of setter for metadata attribute.
        Maybe this is not necessary.
        """
        self.metadata = metadata

    # TODO: consider if the attribute name 'object_metadata'
    # and description 'object level' is clear
    @property
    def object_metadata(self) -> List[pd.DataFrame]:
        """Get the object level metadata for the dataset."""
        return self._object_metadata
    
    @object_metadata.setter
    def object_metadata(self, object_metadata: Optional[List[pd.DataFrame]]=None):

        if object_metadata is None:
            self._object_metadata = [pd.DataFrame()] * len(self.file_index)
            return
                
        if isinstance(object_metadata, List):
            if not all(isinstance(md, pd.DataFrame) for md in object_metadata):
                raise TypeError("All object metadata entries must be pandas DataFrames.")
        else:
            raise TypeError(
                "Expected object_metadata to be a list of pandas DataFrames, "
                f"got {type(object_metadata).__name__} instead."
            )
        
        self._object_metadata = object_metadata

    def set_object_metadata(
            self,
            object_metadata: Optional[List[pd.DataFrame]]=None
        ):
        """
        Wrapper of setter for object_metadata attribute.
        Maybe this is not necessary.
        """
        self.object_metadata = object_metadata

    def update_object_metadata(
            self, 
            metadata: pd.DataFrame, 
            idx: int
        ):
        """
        Update the object level metadata for a specific index.
        This base ImageDataset class does not parse/initialize metadata despite having a placeholder attribute, 
            this method is intended for user custom metadata specification or use by subclasses to set metadata after initialization.
        Overwrites any existing metadata at that index.

        :param metadata: DataFrame containing metadata to set.
        :param idx: Index at which to set the metadata.
        """
        self._check_idx(idx)
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("Metadata must be a pandas DataFrame.")
        self._object_metadata[idx] = metadata

    """
    Dataset Export and Imports
    """

    def _serialize_df(self, df: pd.DataFrame) -> Union[None, Dict]:
        """
        Serialize a DataFrame to a dictionary format for JSON export.
        
        :param df: DataFrame to serialize.
        :return: Dictionary representation of the DataFrame, or None if empty.
        """
        if df is None or df.shape[1] < 1:
            return None
        return df.to_dict(orient='list')

    def export_configuration(self, filepath: str) -> str:
        """
        Export the dataset configuration and current state to a JSON file.
        The JSON file can be logged as a mlflow artifact for reproducibility.
        
        :param filepath (str): Path to the JSON file where the configuration will be saved.
        :return: The absolute path to the saved configuration file. Mostly for mlflow
            artifact logging purposes.
        """

        # TODO this is very flat, unstructured dataset config
        # might need some improvements?
        # TODO also note that currently the transforamtions are not serialized 
        # and will be lost upon re-loading the dataset with `import_dataset`.
        # modification of the transform class will be needed for preservation of transformations
        # across dataset import/export.
        config = {
            "input_channel_keys": self.input_channel_keys,
            "target_channel_keys": self.target_channel_keys,
            "file_index": self.file_index.to_dict(orient='list'),
            # TODO no fail-safe for dataframe serialization here, might need that
            "metadata": self._serialize_df(self._metadata), 
            "object_metadata": {idx: self._serialize_df(md) for idx, md in enumerate(self.object_metadata)}
        }

        filepath = Path(filepath).resolve()
        try:
            with open(filepath, 'w') as file:
                json.dump(config, file, indent=4)
            return str(filepath)
        except IOError as e:
            raise IOError(f"Failed to write configuration to {filepath}: {str(e)}")

    @classmethod
    def load_from_configuration(cls, filepath: str) -> "BaseImageDataset":
        """
        Recreate a dataset object from a JSON configuration file.
        
        :param filepath (str): Path to the JSON file containing the dataset configuration.
        :return: An instance of ImageDataset with the configuration loaded.
        """
        with open(filepath, 'r') as file:
            config = json.load(file)
        
        # Reconstruct the file index and metadata as DataFrames
        file_index = pd.DataFrame(config["file_index"])
        
        metadata = pd.DataFrame(config["metadata"]) if config["metadata"] is not None else None
        dataset = cls(file_index=file_index, metadata=metadata)

        dataset.set_input_channel_keys(config["input_channel_keys"])
        dataset.set_target_channel_keys(config["target_channel_keys"])

        # Restore object level metadata
        for idx, obj_meta in config["object_metadata"].items():
            dataset.object_metadata[int(idx)] = pd.DataFrame(obj_meta) if obj_meta is not None else pd.DataFrame()
        
        return dataset