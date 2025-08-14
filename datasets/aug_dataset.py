from pathlib import Path
from typing import Optional, List, Sequence

import pandas as pd

from .base_dataset import TransformType
from .bbox_dataset import BBoxCropImageDataset
from .bbox_schema import BBoxSchema
from .run_aug import AugRunner, AugPlanConfig
from .real_augments import BaseAugmentation, RotationAug, TranslationAug

class AugmentedBBoxImageDataset(BBoxCropImageDataset):

    def __init__(
        self,
        file_index: pd.DataFrame,
        bbox_annotations: pd.DataFrame,
        augmentations: Sequence[BaseAugmentation] = [
            RotationAug(), TranslationAug()],
        n_augmentations: Optional[int] = None,
        augment_to_n: Optional[int] = None,
        metadata: Optional[pd.DataFrame] = None,
        object_metadata: Optional[List[pd.DataFrame]] = None,
        post_crop_transform: Optional[TransformType] = None,
        bbox_schema: BBoxSchema = BBoxSchema(),
        seed: int = 42,
        augmentation_config: Optional[AugPlanConfig] = None,
        **kwargs
    ):
        
        if not isinstance(file_index, pd.DataFrame):
            raise TypeError("file_index must be a pandas DataFrame.")
        if not isinstance(bbox_annotations, pd.DataFrame):
            raise TypeError("bbox_annotations must be a pandas DataFrame.")
        
        if augment_to_n is not None:
            if isinstance(augment_to_n, int):
                if augment_to_n > len(file_index):
                    n_augmentations = augment_to_n - len(file_index)
                else:
                    raise ValueError(
                        f"If configured, augment_to_n must be greater than the "
                        f"number of original images ({len(file_index)})."
                    )
            else:
                raise TypeError(
                    "augment_to_n must be an integer or None."
                )
        elif n_augmentations is not None:
            if isinstance(n_augmentations, int):
                if n_augmentations <= 0:
                    raise ValueError("n_augmentations must be a positive integer.")
            else:
                raise TypeError("n_augmentations must be an integer or None.")
        else:
            raise ValueError(
                "Either n_augmentations or augment_to_n must be specified."
            )        

        if augmentation_config is None:
            augmentation_config = AugPlanConfig(
                seed=seed
            )
        elif isinstance(augmentation_config, AugPlanConfig):
            pass
        else:
            raise TypeError("augmentation_config must be an instance of AugPlanConfig.")        

        self._augmentations = augmentations

        runner = AugRunner(
            augmentations=augmentations,
            config=augmentation_config,
            schema=bbox_schema
        )

        payloads = {}
        if metadata is not None and isinstance(metadata, pd.DataFrame):
            payloads['metadata'] = metadata

        aug_df, file_index_merged, bbox_merged, payloads_extended = runner.run(
            file_index=file_index,
            bbox_df=bbox_annotations,
            payloads=payloads,
            n_slots=n_augmentations
        )

        self._aug_df = aug_df
        metadata_expanded = payloads_extended.get('metadata', None)

        super().__init__(
            file_index=file_index_merged,
            bbox_annotations=bbox_merged,
            metadata=metadata_expanded,
            object_metadata=None,
            bbox_schema=bbox_schema,
            post_crop_transforms= post_crop_transform,
            **kwargs
        )

    @property
    def augmentations(self) -> Sequence[BaseAugmentation]:
        return self._augmentations
    
    @property
    def augmentation_df(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing augmentation metadata.
        """
        return self._aug_df
    
    @classmethod
    def from_dataset(
        cls,
        dataset: BBoxCropImageDataset,
        **kwargs
    ) -> "AugmentedBBoxImageDataset":
        """
        Creates an AugmentedBBoxImageDataset from an existing BBoxCropImageDataset.
        """
        if not isinstance(dataset, BBoxCropImageDataset):
            raise TypeError("dataset must be an instance of BBoxCropImageDataset.")
        
        if 'file_index' in kwargs:
            raise ValueError("file_index cannot be overridden; it is derived from the dataset.")
        if 'bbox_annotations' in kwargs:
            raise ValueError("bbox_annotations cannot be overridden; it is derived from the dataset.")
        if 'bbox_schema' in kwargs:
            raise ValueError("bbox_schema cannot be overridden; it is derived from the dataset.")
        
        return cls(
            file_index=dataset.file_index.copy(),
            bbox_annotations=dataset.bbox_df.copy(),
            metadata=dataset.metadata.copy() if \
                dataset.metadata is not None else None,
            object_metadata=dataset.object_metadata.copy() if \
                dataset.object_metadata is not None else None,
            bbox_schema=dataset.bbox_schema,
            **kwargs
        )