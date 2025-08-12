# `datasets` Module — Image Dataset Infrastructure

This module provides a **base class** and supporting **state/view classes** for building image datasets with:

- **Dynamic lazy loading** of input/target images from disk
- **Configurable caching** for reusing decoded image arrays
- **Channel key flexibility** without wasteful reloads
- **Minimal coupling** between file loading, dataset indexing, and metadata management

The goal is to make dataset implementations **fast, memory-efficient, and maintainable**.

---

## **Dataset Backbone**
### 1. `DatasetView`
- **Immutable** object containing:
  - `file_index` — Pandas DataFrame mapping dataset indices to file paths (columns = channels).
  - `pil_image_mode` — Fixed Pillow image mode for decoding.
- Provides:
  - `__len__()` — Number of samples.
  - `get_paths_for_keys(idx, keys)` — Retrieve file paths for specific channels at an index.
  - `read_image(path)` — Read and convert a single image to the configured mode.

**Invariants**:
- `file_index` is validated at init:
  - Non-empty DataFrame
  - All entries are `pathlib.Path`, `PurePath`, or `str`
- `pil_image_mode` is validated against Pillow’s supported modes.
- Both attributes are **fixed** after initialization.

### 2. `IndexState`
- Tracks the **last accessed dataset index**.
- Very lightweight — just `last: Optional[int]`.
- Used as memory of last loaded image intended to allow dataset class to 
return the metadata entry associated with last loaded view/fov.

### 3. `FileState`
- Manages **lazy loading and caching** of decoded image arrays for the current dataset index.
- Given `idx`, `input_keys`, and `target_keys`, it:
  1. Queries `DatasetView` for file paths.
  2. Checks cache for already-decoded images.
  3. Loads only missing channels.
  4. Reorders channels if keys have changed.
  5. Drops unused channels from the current realized stack but keeps them in the cache for possible reuse.

**Cache Behavior**:
- **If `cache_capacity` is `None`**:
  - Defaults to `len(file_index.columns)` — enough to store all channels for one row.
- **If `cache_capacity` is an integer > 0**:
  - Cache acts as an LRU (Least Recently Used) store.
  - When adding a new path:
    - If over capacity, the least recently used path is evicted.
  - Ensures cache memory use is bounded.
- **If `cache_capacity` is `-1`**:
  - Cache is unbounded — stores every decoded image path forever (until manually cleared).

---

## **`BaseImageDataset` Class**

### Purpose
- Provides a **drop-in torch `torch.utils.data.Dataset`** implementation,
compatible with torch `DataLoader`.
  - **Access of images** via `obj[i]`
- Serves as the foundation for subclass dataset implementation.
- Encapsulates:
  - **File access** via `DatasetView` + `FileState`
  - **Index tracking** via `IndexState`
  - **Optional metadata** handling (`metadata`, `object_metadata`)
- Allows for configuration of at and following construction:
  - **Input/Target channels**
  - **Albumentations transforms** (`transform`, `input_only_transform`, `target_only_transform`)

---

### Key Features

- **Immutable file mapping**:  
  `file_index` and `pil_image_mode` are fixed for the lifetime of the dataset.

- **Channel keys setters allowing for re-configuring input/targets**:  
  Changing `input_channel_keys` or `target_channel_keys` will not clear state — `FileState` reuses any matching cached images.

- **Optional metadata**:  
  - `BaseImageDataset` holds optionally `metadata` (per-row DataFrame) — 
  must match length of `file_index`.
  - Likewise holds `object_metadata` (list of DataFrames) — 
  must match length of `file_index`.
  - Intended for subclass usage where metadata/object level metadata can alter
  dataset behavior.  

- **Transforms**:  
  - `transform`: applied to both input & target simultaneously (with `target` as additional target).
  - `input_only_transform`: applied only to input, following `transform`.
  - `target_only_transform`: applied only to target, following `transform`.

- **JSON serialization** support: 
  - `to_json_config(filepath)` for saving dataset configurations as human readable json. Note
  that currently the transform configuration will be lost due to inability to serialize transform
  objects.
  - `from_json_config(filepath, **transforms)` for loading dataset configurations back.
---

### Construction Example
```python
dataset = BaseImageDataset(
    file_index: pd.DataFrame,
    pil_image_mode: str = "I;16",
    metadata: Optional[pd.DataFrame] = None,
    object_metadata: Optional[Sequence[pd.DataFrame]] = None,
    input_channel_keys: Optional[Union[str, Sequence[str]]] = None,
    target_channel_keys: Optional[Union[str, Sequence[str]]] = None,
    transform: Optional[TransformType] = None,
    input_only_transform: Optional[TransformType] = None,
    target_only_transform: Optional[TransformType] = None,
    cache_capacity: Optional[int] = None
)
```

### Save/Load Example

```python

dataset.to_json_config("dataset_config.json")

dataset_loaded = BaseImageDataset.from_json_config(
    "dataset_config.json",
    transform=augmentation_pipeline,
    input_only_transform=input_preprocessing,
    target_only_transform=target_preprocessing
)
```