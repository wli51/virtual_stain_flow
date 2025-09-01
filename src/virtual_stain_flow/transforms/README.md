# `transforms` Module — Image Transformation Utilities

This module provides a collection of **image transformation utilities** designed for preprocessing and augmenting image datasets. It includes:

- **Base class for custom transforms** with serialization support
- **Normalization transforms** for scaling and standardizing image data
- **Validation utilities** for ensuring compatibility with Albumentations pipelines

The goal is to provide a flexible and extensible framework for applying transformations to images in a reproducible and configurable manner.

---

## **Transform Backbone**

### 1. `BaseTransform`
- Abstract base class for creating custom image transformations.
- Inherits from `albumentations.ImageOnlyTransform` to integrate seamlessly with Albumentations pipelines.
- Provides:
    - **Serialization support**: Enables saving and loading transform configurations as dictionaries.
    - **Custom transformation logic**: Subclasses implement the `apply` method to define transformation behavior.
    - **Reproducibility**: Includes a `name` attribute for identifying transforms.

**Key Features**:
- `apply(img, **params)` — Abstract method for applying the transformation to an image.
- `serialize()` — Converts the transform configuration into a dictionary for export.
- `from_config(config)` — Class method for creating a transform instance from a serialized configuration.

---

## **Normalization Transforms**
Currently there are only two realizations of the transform class, both of which 
are for normalization purposes, they are:

### 1. `MaxScaleNormalize`
- Scales image pixel values to the range `[0, 1]` based on a specified normalization factor.
- Supports:
    - Custom normalization factors (e.g., `65535` for 16-bit images, `255` for 8-bit images).
    - Literal values `'16bit'` and `'8bit'` for convenience.

**Constructor Parameters**:
- `normalization_factor` — Factor used to scale pixel values (e.g., `65535` for 16-bit images).
- `name` — Name of the transform (default: `"MaxScaleNormalize"`).
- `p` — Probability of applying the transform (default: `1.0`).

### 2. `ZScoreNormalize`
- Normalizes images to have a mean of `0` and a standard deviation of `1`.
- Supports:
    - Per-image normalization (default behavior).
    - Pre-specified mean and standard deviation values.

**Constructor Parameters**:
- `mean` — Pre-specified mean value (optional).
- `std` — Pre-specified standard deviation value (optional).
- `name` — Name of the transform (default: `"ZScoreNormalize"`).
- `p` — Probability of applying the transform (default: `1.0`).

---

## **Validation Utilities**
Utilities for validating legal transformations is centralized in this module.
Allowed transformation types include subclasses of `BaseTransform` as well as
`ImageOnlyTransform`, `BasicTransform` and `Compose` from `albumentations`.
Only subclasses of `BaseTransform` support serilization and logging. 

### 1. `is_valid_image_transform`
- Checks if an object is a valid Albumentations image transform.
- Returns `True` if the transform targets images.

### 2. `validate_compose_transform`
- Validates and returns a `Compose` transform.
- Ensures compatibility with Albumentations pipelines and supports additional targets (e.g., `target` images).

**Constructor Parameters**:
- `obj` — Transform object to validate (e.g., `Compose`, `ImageOnlyTransform`, or a sequence of transforms).
- `apply_to_target` — Whether the transform should be applied to target images (default: `True`).

**Usage Example**:
```python
from albumentations import HorizontalFlip

transform = validate_compose_transform(HorizontalFlip(p=0.5))
```

---

## **Serialization and Loading**

### Save/Load Example
```python
# Serialize a transform
transform = MaxScaleNormalize(normalization_factor="16bit")
config = transform.serialize()

# Load a transform from configuration
loaded_transform = load_transform(config)
```

---

## **Key Features (shared across all transforms)**

- **Albumentations compatibility**:  
    All transforms integrate seamlessly with Albumentations pipelines.

- **Serialization support**:  
    Transforms can be serialized to dictionaries and reloaded for reproducibility.

- **Customizability**:  
    Users can extend `BaseTransform` to create their own transformations.

- **Validation utilities**:  
    Ensures transforms are compatible with image processing pipelines.

---

## **Example Pipeline Integration**
```python
from albumentations import Compose
from transforms.normalizations import MaxScaleNormalize, ZScoreNormalize

# Define a transformation pipeline
pipeline = Compose([
        MaxScaleNormalize(normalization_factor="16bit"),
        ZScoreNormalize()
])

# Apply the pipeline to an image
transformed_img = pipeline(image=img)["image"]