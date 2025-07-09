from .BaseImageDataset import BaseImageDataset
from .CPLoadDataImageDataset import CPLoadDataImageDataset

# type aliases for type hinting
ImageDatasetType = BaseImageDataset # base class and everything that inherits from it
ImageDatasetTypeRuntime = (BaseImageDataset,)

__all__ = [
    "ImageDatasetType",
    "ImageDatasetTypeRuntime",
    "BaseImageDataset",
    "CPLoadDataImageDataset",
]