import random
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

from .PatchCollection import PatchCollection

class ImagePatcher:
    """
    Helper class for generating patches from an single image based on optional 
        object coordinate metadata. 
    Two supported methods for patch generation include:
        1. Randomly sampling patches from the image.
        2. Sampling patches centered on randomly selected object coordinates with optional jitter.
    Uses PatchCollection instance and curates object level metadata for generated patches.    
    """
    def __init__(
            self,
            obj_coordinates: pd.DataFrame,
            patch_collection: Optional[PatchCollection] = None,
            obj_metadata: Optional[pd.DataFrame] = None,
            image_shape: tuple[int, int] = (1024, 1024),
            patch_size: int = 256,
            # TODO modify this class to accept rngs intead of random_seed
            random_seed: Optional[int] = None
        ):
        """
        PatchGenerator initializes with a PatchCollection and image dimensions.

        :param obj_coordinates: pd.DataFrame with object coordinates (x, y)
        :param patch_collection: Optional PatchCollection instance, if None a new one is created
        :param obj_metadata: Optional pd.DataFrame with metadata for each object
        :param image_shape: tuple of (height, width) for the image
        :param patch_size: int, size of the square patches to generate
        :param random_seed: Optional int, random seed for reproducibility
        """
        if patch_collection is None:
            if obj_coordinates is None:
                raise ValueError("obj_coordinates must be provided")
            patch_collection = PatchCollection(obj_coordinates=obj_coordinates)
        elif not isinstance(patch_collection, PatchCollection):
            raise TypeError("patch_collection must be an instance of PatchCollection")
        else:
            # ensure the patch collection has the correct object coordinates
            patch_collection._obj_coordinates = obj_coordinates

        self.pc = patch_collection

        if obj_metadata is None:
            self._obj_metadata = obj_coordinates
        elif not isinstance(obj_metadata, pd.DataFrame):
            raise TypeError("obj_metadata must be a pandas DataFrame")
        elif len(obj_metadata) != len(obj_coordinates):
            raise ValueError("obj_metadata must have the same number of rows as obj_coordinates")
        else:
            self._obj_metadata = obj_metadata
            
        # curated object level metadata will be update as patches are generated
        # this ensures that all patches will only have object level metadata
        # corresponding only to captured objects
        self._cur_obj_metadata: List[pd.DataFrame] = []

        if not isinstance(image_shape, tuple) or len(image_shape) != 2:
            raise ValueError("image_shape must be a tuple of (height, width)")        
        self.h, self.w = image_shape

        if not isinstance(patch_size, int):
            raise TypeError("patch_size must be an integer")
        elif patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self._patch_size = patch_size

        if (not random_seed is None) and (not isinstance(random_seed, int)):
            raise TypeError("random_seed must be an integer")
        self.__rng = random.Random(random_seed)
        self.__np_rng = np.random.default_rng(random_seed)

    def random_patches(
            self,
            num_patches, 
            max_attempts=1000, 
            allow_overlap=False
        ):
        """
        Add randomly sampled patches to the PatchCollection.

        :param num_patches: int, number of patches to generate
        :param max_attempts: int, limit for rejection sampling
        :param allow_overlap: bool
        """
        attempts = 0
        while len(self.pc) < num_patches and attempts < max_attempts:
            x = self.__rng.randint(0, self.w - self._patch_size)
            y = self.__rng.randint(0, self.h - self._patch_size)
            box = [x, y, x + self._patch_size, y + self._patch_size]

            if allow_overlap or not self.pc.check_overlap(box):
                self.pc.add(box)
            else:
                attempts += 1

        self.curate_obj_metadata()

    def cell_containing_patches(
            self,
            num_patches, 
            allow_overlap=False, 
            max_overlap_ratio: Optional[float] = None,
            jitter_ratio=0.25,
            max_attempts=1000,
            early_stop=True
        ):
        """
        Add patches centered (with jitter) on randomly selected object coordinates.

        :param num_patches: int
        :param allow_overlap: bool
        :param max_overlap_ratio: float, maximum allowed overlap ratio with existing patches
        :param jitter_ratio: float, ratio of patch size used as jitter window
        :param max_attempts: int
        :param early_stop: bool, stop if all objects are captured
        """
        obj_coords = self.pc._obj_coordinates
        if obj_coords is None or obj_coords.empty:
            return
        
        if max_overlap_ratio is None:
            pass
        elif isinstance(max_overlap_ratio, (int, float)):
            if max_overlap_ratio < 0 or max_overlap_ratio > 1:
                raise ValueError("max_overlap_ratio must be between 0 and 1")
        else:
            raise TypeError("max_overlap_ratio must be a float or None")

        def overlap_check_fn(box):
            """
            Check if the box overlaps with existing patches.
            If max_overlap_ratio is specified, check for soft overlap.
            """
            if max_overlap_ratio is None:
                return self.pc.check_overlap(box)
            else:
                return self.pc.check_overlap_soft(box, threshold=max_overlap_ratio)

        attempts = 0
        while len(self.pc) < num_patches and attempts < max_attempts:
            row = obj_coords.sample(n=1, random_state=self.__np_rng.integers(2**32)).iloc[0]
            x, y = row.iloc[0], row.iloc[1]

            offset_range = int(self._patch_size * jitter_ratio)
            dx = self.__rng.randint(-offset_range, offset_range)
            dy = self.__rng.randint(-offset_range, offset_range)

            x = min(max(x - self._patch_size // 2 + dx, 0), self.w - self._patch_size)
            y = min(max(y - self._patch_size // 2 + dy, 0), self.h - self._patch_size)
            box = [x, y, x + self._patch_size, y + self._patch_size]

            if allow_overlap or not overlap_check_fn(box):
                self.pc.add(box)
                self.pc.update_captured_objects()
            else:
                attempts += 1

            if early_stop and all(self.pc.obj_captured_flag):
                break

        self.curate_obj_metadata()

    def curate_obj_metadata(self):
        """
        Curate the object metadata to match the captured objects in the PatchCollection.

        :return: pd.DataFrame with metadata for captured objects
        """
        for i in range(len(self._cur_obj_metadata), len(self.pc)):
            contained_obj_ids = []
            for j, row in self.pc._obj_coordinates.iterrows():
                x, y = row.iloc[0], row.iloc[1]
                if self.pc.contains_point(
                    index=i,
                    x=x,
                    y=y
                ):
                    contained_obj_ids.append(row.name)
            if len(contained_obj_ids) == 0:
                self._cur_obj_metadata.append(pd.DataFrame(columns=self._obj_metadata.columns))
            else:
                self._cur_obj_metadata.append(
                    self._obj_metadata.iloc[contained_obj_ids,:]
                )

    @property
    def patch_collection(self) -> PatchCollection:
        """
        Returns the PatchCollection instance.

        :return: PatchCollection
        """
        return self.pc
    
    @property
    def curated_object_metadata(self) -> List[pd.DataFrame]:
        """
        Returns the curated object metadata for each patch.

        :return: List of pd.DataFrame, one for each patch
        """
        return self._cur_obj_metadata
    
class PatchGenerator:
    """
    A simple wrapper class for pre-configuring and running patch generation methods.
    Allows for more flexible and reusable patch generation pipelines as a sequence 
        of steps, each step being a method of ImagePatcher with its own parameters.
    """

    def __init__(
            self, 
            steps: List[Tuple[str, Dict]], 
            patch_size: int, 
            seed: Optional[int] = None
        ):
        """
        PatchGenerator is a configurable patch generation pipeline.

        :param steps: List of (method_name, kwargs) tuples, e.g.
                      [("cell_containing_patches", {...}), ("random_patches", {...})]
                      See ImagePatcher methods for available methods and their parameters.
        :param patch_size: int, size of the square patches to generate
        :param seed: optional random seed for reproducibility
        """        

        # type check for steps
        if not isinstance(steps, list):
            raise TypeError("steps must be a list of tuples (method_name, kwargs)")
        for step in steps:
            
            if not isinstance(step, tuple):
                raise TypeError(f"Expected each step to be a tuple (method_name, kwargs), got {type(step)}")
            if len(step) != 2:
                raise ValueError(f"Each step must be a tuple of length 2, got {len(step)}")
            
            method_name, kwargs = step
            
            if not isinstance(method_name, str):
                raise TypeError(f"Expected method_name to be a string, got {type(method_name)}")
            if not hasattr(ImagePatcher, method_name):
                raise AttributeError(f"ImagePatcher has no method '{method_name}'"
                                     "Available methods: random_patches, cell_containing_patches")
            if not isinstance(kwargs, dict):
                raise TypeError(f"Expected kwargs to be a dictionary, got {type(kwargs)}")
            # kwargs check at runtime            

        self.steps = steps

        # TODO move type checks from ImagePatcher to here?
        self.patch_size = patch_size
        self.seed = seed

        # TODO pass these into ImagePatcher
        self.__rng = random.Random(seed),
        self.__np_rng = np.random.default_rng(seed)

    def __call__(
            self, 
            image_shape: Tuple[int, int], 
            obj_coordinates: pd.DataFrame,
            obj_metadata: Optional[pd.DataFrame] = None
        ):
        """
        Generate a PatchCollection by applying the configured steps.

        :param image_shape: (height, width) tuple
        :param obj_coordinates: DataFrame with at least two columns (x, y)
        :return: PatchCollection
        """

        backend = ImagePatcher(
            obj_coordinates = obj_coordinates,
            obj_metadata = obj_metadata, 
            image_shape = image_shape,
            patch_size=self.patch_size,
            # TODO modify ImagePatcher to accept rngs intead of random_seed
            random_seed = self.seed,
        )
        try:
            for method_name, kwargs in self.steps:
                method = getattr(backend, method_name)
                method(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error running method {method_name} during patch generation: {e}")
        
        return backend.patch_collection, backend.curated_object_metadata