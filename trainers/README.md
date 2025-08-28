# `virtual_stain_flow.trainers`
## Overview

This module provides **trainer classes** for training and evaluating models, including FNet/UNet and WGAN-GP architectures. The trainers are designed with modularity and extensibility in mind, leveraging abstract base classes to isolate shared components.

The `LoggingTrainer` and `LoggingGANTrainer` classes are the primary trainers that should be used for training models. The `AbstractTrainer` and `AbstractLoggingTrainer` classes serve as base classes for abstraction purposes and should not be used directly.

---

## Module Structure
```bash
virtual_stain_flow/
└── trainers/
    ├── logging_trainers/
    |   ├── AbstractLoggingTrainer.py
    |   ├── LoggingTrainer.py
    |   └── LoggingGANTrainer.py
    ├── AbstractTrainer.py
    └── README.md
```

---

## **Trainer Classes**

### 1. `AbstractTrainer`
- Serves as the base class for all trainers.
- Provides shared functionality such as dataset handling, early stopping, and loss/metric tracking.
- Implements abstract methods like `train_step`, `evaluate_step`, `train_epoch`, and `evaluate_epoch` that must be overridden by subclasses.
---

### 2. `AbstractLoggingTrainer`
- Extends `AbstractTrainer` to add support for logging training progress using MLflow.
- Provides hooks for logging metrics, losses, and model checkpoints.
- Implements a reworked `train` method to interface with the logger.
- Abstract `log_model` method for logging model configurations.

---

### 3. `LoggingTrainer`
- A concrete implementation of `AbstractLoggingTrainer` for single-network models like FNet/UNet.
- Supports multiple loss functions with customizable weights.
- Handles training and evaluation of the model, including metric computation and logging.

**Usage Example**:
```python
from trainers.LoggingTrainer import LoggingTrainer

trainer = LoggingTrainer(
    model=model,
    optimizer=optimizer,
    backprop_loss=loss_fn,
    dataset=dataset,
    batch_size=16,
    train_for_epochs=10,
    patience=5,
    metrics={"PSNR": psnr_metric, "SSIM": ssim_metric},
)
trainer.train(logger=mlflow_logger)
```

---

### 4. `LoggingGANTrainer`
- A specialized trainer for training GANs, including WGAN-GP.
- Manages both the generator and discriminator, with separate optimizers and loss functions.
- Supports gradient penalty and adversarial loss for stable GAN training.

**Usage Example**:
```python
from trainers.LoggingGANTrainer import LoggingGANTrainer

trainer = LoggingGANTrainer(
    generator=generator,
    generator_optimizer=gen_optimizer,
    generator_reconstruction_loss_fn=[recon_loss_fn],
    discriminator=discriminator,
    discriminator_optimizer=disc_optimizer,
    dataset=dataset,
    batch_size=16,
    train_for_epochs=10,
    patience=5,
    metrics={"PSNR": psnr_metric, "SSIM": ssim_metric},
)
trainer.train(logger=mlflow_logger)
```