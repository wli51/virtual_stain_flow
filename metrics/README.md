Here lives the metric classes which is dependent on a abstract metric class
Each metric needs to have a foward function implemented over target and predict while the abstract class functions inhertied handles accumulation
## Overview

This module provides a collection of **metric classes** for evaluating the performance of models. Metrics are essential for assessing the quality of predictions and guiding model improvements. The metrics in this module are built on an abstract base class, ensuring consistency and extensibility.

---

## Module Structure
```bash
virtual_stain_flow/
└── metrics/
    ├── AbstractMetrics.py
    ├── MetricsWrapper.py
    ├── PSNR.py
    ├── SSIM.py
    └── README.md
```

---

## **Metric Classes**

### 1. `AbstractMetrics`
- Serves as the base class for all metrics.
- Provides functionality for accumulating metric values across batches and aggregating them using methods like `mean` or `sum`.

**Key Features**:
- `update()` — Updates the metric values for training or validation data.
- `reset()` — Resets the accumulated metric values.
- `compute()` — Computes the aggregated metric value.

---

### 2. `PSNR`
- Computes the **Peak Signal-to-Noise Ratio (PSNR)**, a common metric for image quality assessment.
- Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.

**Constructor Parameters**:
- `_metric_name` — Name of the metric.
- `_max_pixel_value` — Maximum possible pixel value of the images (default: `1`).

**Usage Example**:
```python
from metrics.PSNR import PSNR

psnr_metric = PSNR("PSNR")
psnr_value = psnr_metric(generated_images, target_images)
```

---

### 3. `SSIM`
- Computes the **Structural Similarity Index Measure (SSIM)**, which evaluates the similarity between two images.
- Considers luminance, contrast, and structure for a more perceptually relevant assessment.

**Constructor Parameters**:
- `_metric_name` — Name of the metric.
- `_max_pixel_value` — Maximum possible pixel value of the images (default: `1`).

**Usage Example**:
```python
from metrics.SSIM import SSIM

ssim_metric = SSIM("SSIM")
ssim_value = ssim_metric(generated_images, target_images)
```

---

### 4. `MetricsWrapper`
- Wraps a PyTorch module to compute and accumulate metric values across batches.
- Useful for integrating custom metrics implemented as PyTorch modules.

**Constructor Parameters**:
- `_metric_name` — Name of the metric.
- `module` — A PyTorch module with a `forward` function.

**Usage Example**:
```python
from metrics.MetricsWrapper import MetricsWrapper
from some_module import CustomMetricModule

custom_metric = MetricsWrapper("CustomMetric", CustomMetricModule())
metric_value = custom_metric(generated_images, target_images)
```

---