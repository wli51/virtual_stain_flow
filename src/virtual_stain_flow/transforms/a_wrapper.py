"""
/transforms/a_wrapper.py

This module defines a wrapper class for Albumentations transforms, allowing
serialization and deserialization of Albumentations transforms to better
integrate with logging. 

ALbumentations comes with its own serialization
mechanism which does not always deserialize back correctly. 
This wrapper utilizes the more robust .to_dict() method from Albumentations
for serialization and a custom deserialization function to reconstruct the
Albumentations transform from the dictionary representation.

Classes:
- AWrapper
"""

from typing import Dict

from .transform_utils import (
    ValidAlbumentationType, RuntimeValidAlbumentationType
)
from .base_transform import LoggableTransform

class AWrapper(LoggableTransform):

    def __init__(
        self,
        obj: ValidAlbumentationType
    ):
        
        if isinstance(obj, LoggableTransform):
            raise TypeError(
                "The provided object is already a LoggableTransform."
            )
        elif isinstance(obj, RuntimeValidAlbumentationType):
            pass
        else:
            raise TypeError(
                "The provided object is not a valid Albumentations transform."
            )

        class_name = type(obj).__name__ 
        # Special handling for to include child transforms if applicable
        # in the wrapper name for more expressive representation.
        child_transforms = getattr(obj, 'transforms', [])
        child_names = [
            type(t).__name__ for t in child_transforms
        ]
        name = f'{class_name}(' + ','.join(child_names) + ')'
        super().__init__(
            name = name,
            p = obj.p
        )
        self.transform = obj
        self.config = {
            'class': 'AWrapper',
            'name': name,
            'params': obj.to_dict()
        }

    def apply(self, img, **params):
        """
        Simply delegates to the wrapped Albumentations transform's apply method.
        """
        return self.transform.apply(img, **params)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(name={self.name}, "
                f"p={self.p})"
        )
    
    def to_config(self, **kwargs) -> Dict:
        return self.config

    @classmethod
    def from_config(cls, config: Dict) -> 'LoggableTransform':
        if config.get('class', None) != 'AWrapper':
            raise ValueError(
                "The provided config does not correspond to an AWrapper "
                "transform."
            )

        if 'params' not in config:
            raise ValueError(
                "The provided config does not have expected 'params' key."
            )

        return cls(_deserialize_albumentation(config['params']))                

def _deserialize_albumentation(
    config: Dict
) -> ValidAlbumentationType:
    if 'transform' in config:
        config = config['transform']
    
    import importlib
    albumentations_module = importlib.import_module("albumentations")
    class_name = config.pop('__class_fullname__', None)
    if class_name is None:
        raise ValueError(
            "The provided Albumentations transform does not have expected "
            "__class_fullname__ attribute."
        )
    try:
        transform_cls = getattr(albumentations_module, class_name)
    except AttributeError:
        raise ValueError(
            f"The provided Albumentations transform class name '{class_name}' "
            "is not valid."
        )
    if class_name == 'Compose':
        transform_configs = config.pop('transforms', [])
        transforms = [
            _deserialize_albumentation({'transform': t}) for t in transform_configs
        ]
        config['transforms'] = transforms

    try:
        obj = transform_cls(**config)
    except Exception as e:
        raise RuntimeError(
            f"The provided Albumentations transform class name '{class_name}' "
            "could not be instantiated with the provided configuration."
        ) from e
    
    return obj
