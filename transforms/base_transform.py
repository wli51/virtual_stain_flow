# /transforms/base_transform.py

from abc import ABC, abstractmethod

from albumentations import ImageOnlyTransform
import numpy as np

"""
Base class for image transforms using Albumentations.
Serializable transform class for (albumentation based) image transformations,
facilitating easy logging. 
"""
class BaseTransform(ABC, ImageOnlyTransform):
    
    def __init__(
            self, 
            name: str,
            p: float = 1.0,
            **kwargs
        ):

        self._name = name

        super().__init__(
            p=p,
            **kwargs
        )

    @property
    def name(self) -> str:
        """
        Get the name of the transform.
        
        :return: Name of the transform.
        """
        return self._name

    @abstractmethod
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Apply the transformation to the input image.
        
        :param img: Input image as a NumPy array.
        :return: Transformed image as a NumPy array.
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the transform.
        
        :return: String representation of the transform.
        """
        return (f"{self.__class__.__name__}(name={self._name}, "
                f"p={self.p})"
        )
    
    @abstractmethod
    def serialize(
            self, 
            **kwargs
        ) -> dict:
        """
        Serialize the transform to a dictionary exportable to YML/JSON.

        Should create a dictionary with:
        - 'class': The class name of the transform.
        - 'name': The name of the transform.
        - 'params': A dictionary of parameters specific to the transform.
        """
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            name=config['name'],
            **config['params']
        )
    
def load_transform(
        config: dict
    ) -> BaseTransform:
    """
    Load a transform from a configuration dictionary.
    
    :param config: Configuration dictionary containing the class name and parameters.
    :return: An instance of the transform class.
    """
    
    cls_name = config["class"]
    try:
        cls = globals()[cls_name]
    except KeyError:
        raise ValueError(
            f"Transform class '{cls_name}' not found in globals. "
            "Please ensure the class is defined and imported correctly."
        )
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while loading the transform class '{cls_name}': {e}"
        )
    
    if callable(getattr(cls, 'from_config')):
        return cls.from_config(config)
    else:    
        return cls(
            name=config['name'],
            **config['params']
        )