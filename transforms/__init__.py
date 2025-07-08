from typing import Union, Optional

from albumentations import ImageOnlyTransform
from albumentations.core.composition import Compose

from .MinMaxNormalize import MinMaxNormalize
from .PixelDepthTransform import PixelDepthTransform
from .ZScoreNormalize import ZScoreNormalize

TransformType = Optional[Union[ImageOnlyTransform, Compose]]
# Runtime check type tuple
TransformTypeRuntime = (ImageOnlyTransform, Compose, type(None))
transform_type_error_text = "Transform must be an instance of "
"albumentations.ImageOnlyTransform or albumentations.Compose or None. "

__all__ = [
    "TransformType",
    "TransformTypeRuntime",
    "transform_type_error_text",
    "MinMaxNormalize",
    "PixelDepthTransform",
    "ZScoreNormalize"
]