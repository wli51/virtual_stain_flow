from typing import Union, Sequence

from albumentations import Compose, ImageOnlyTransform, BasicTransform

from .base_transform import BaseTransform
from .normalizations import MaxScaleNormalize, ZScoreNormalize

TransformType = Union[BasicTransform, ImageOnlyTransform, Compose]

def is_valid_image_transform(
    obj: TransformType,
) -> bool:
    if isinstance(obj, (BasicTransform, ImageOnlyTransform)):
        return 'image' in obj.targets
    else:
        return False    

def validate_compose_transform(
    obj: TransformType,
    apply_to_target: bool = True,
) -> Compose:
    """
    Validates and returns a Compose transform.
    
    :param obj: The transform object to validate.
    :param apply_to_target: Whether the transform should be applied to 
        target images.
    :return: Validated Compose transform.
    :raises TypeError: If the transform is not valid.
    """
    add_targets = {'target': 'image'} if apply_to_target else {}

    if is_valid_image_transform(obj):
        return Compose([obj], additional_targets=add_targets)

    elif isinstance(obj, Sequence):
        if not all(is_valid_image_transform(t) for t in obj):
            raise TypeError("All items must be ImageOnlyTransform instances.")
        return Compose(list(obj), additional_targets=add_targets)

    elif isinstance(obj, Compose):
        if apply_to_target and 'target' not in getattr(
            obj, 'additional_targets', {}):
            raise ValueError(
                "apply_to_target=True requires 'target' in "
                "Compose.additional_targets."
            )
        return obj
    else:
        raise TypeError(
            "Expected Compose, ImageOnlyTransform, or "
            f"Sequence[ImageOnlyTransform], got {type(obj)}."
        )
    
__all__ = [
    "is_valid_image_transform",
    "validate_compose_transform",
    "BaseTransform",
    "MaxScaleNormalize",
    "ZScoreNormalize",
]