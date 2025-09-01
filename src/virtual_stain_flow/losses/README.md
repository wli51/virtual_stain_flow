# `virtual_stain_flow.losses`
## Overview

This module provides a collection of special **loss functions** for virtual staining model training. Generic error losses such as the `L1Loss()` from `torch` can be directly used to train simple models. 

The module contains a loss collection for Wasserstein Generative Adversarial Networks (wGAN) with Gradient Penalty (wGAN-GP). It includes:

- **Discriminator loss** for training the discriminator to distinguish real from generated data.
- **Generator loss** for training the generator to produce realistic data.
- **Gradient penalty loss** for enforcing the Lipschitz constraint on the discriminator.

The goal is to provide a modular and extensible framework for implementing and customizing loss functions in GAN training.

---

## Module Structure
```bash
virtual_stain_flow/
└── losses/
    ├── AbstractLoss.py
    ├── loss_item_group.py
    ├── wgan_losses.py
    └── README.md
```
---

## **Loss Functions**

### 1. `WassersteinDiscriminatorLoss`
- Implements the Wasserstein loss for the discriminator.
- Encourages the discriminator to assign higher scores to real data and lower scores to generated data.

**Constructor Parameters**:
- `metric_name` — Name of the loss (default: `"WassersteinDiscriminatorLoss"`).

**Usage Example**:
```python
from losses.wgan_losses import WassersteinDiscriminatorLoss

loss_fn = WassersteinDiscriminatorLoss()
loss = loss_fn(expected_real_as_real_prob, expected_fake_as_real_prob)
```

---

### 2. `GradientPenaltyLoss`
- Enforces the Lipschitz constraint by penalizing the gradient norm of the discriminator's output with respect to interpolated inputs.

**Constructor Parameters**:
- `metric_name` — Name of the loss (default: `"GradientPenaltyLoss"`).

**Usage Example**:
```python
from losses.wgan_losses import GradientPenaltyLoss

loss_fn = GradientPenaltyLoss()
loss = loss_fn(real_target_input_stack, fake_target_input_stack, discriminator)
```

---

### 3. `AdversarialGeneratorLoss`
- Encourages the generator to produce data that the discriminator classifies as real.

**Constructor Parameters**:
- `metric_name` — Name of the loss (default: `"AdversarialGeneratorLoss"`).

**Usage Example**:
```python
from losses.wgan_losses import AdversarialGeneratorLoss

loss_fn = AdversarialGeneratorLoss()
loss = loss_fn(expected_fake_as_real_prob)
```

---

## **(Internal) Loss Item and Group Utilities**

### 1. `LossItem`
- Wraps a loss function with additional metadata, such as a key, weight, and computation phase.

**Key Features**:
- `compute(ctx)` — Computes the raw and weighted loss values.
- `active(phase)` — Checks if the loss is active for a given phase (`'train'` or `'eval'`).

### 2. `LossGroup`
- A collection of `LossItem` objects for managing multiple related losses (dependent on a collection of model predictions in a single forward pass) in a training pipeline. Currently the main purpose of `LossGroup` is to abstract out the separate loss computation and model update logic surrounding wGAN generator/discriminator training. 

**Key Features**:
- `__call__(ctx, phase)` — Computes the total loss and logs for all active loss items.
- `set_weights(weights)` — Updates the weights of loss items.
- `enable(keys)` — Enables specific loss items by their keys.

---

## **Internal Use of `LossItem` and `LossGroup` in wGAN training**
Currently this is all internally defined used in trainer, user does not need to worry about this:
```python
# imports ...

# In wGAN training, the generator and discriminator updates are sparate and involves different forward passes. 

# This loss group bundles up all losses needed for
# generator training:
gen_loss_group = LossGroup(
    "generator_losses", # name, can be anything logical
    [
        *[LossItem(
            module=loss_fn, 
            args=("target", "pred"), # says what inputs are needed to compute these losses
            weight=weight)
            for loss_fn, weight in zip(generator_reconstruction_loss_fn, generator_reconstruction_loss_weights)], # a list of standard ML loss functions taking model prediction and ground truth targets and returning a scalar, wrapped in LossItems, as args
        LossItem(
            module=generator_adversarial_loss_fn,
            args=("discriminator_fake_as_real_prob",),
            weight=1.0) # special adveserial loss requiring a specific forward pass intermediate prdocued when running forward pass first over generator and then into the discriminator (detached) 
    ]
)

# THis loss group bundles up all losses needed for discriminator trianing:
disc_loss_group = LossGroup(
    "discriminator_losses", # logical name
    [
        LossItem(
            module=discriminator_loss_fn,
            args=("discriminator_real_as_real_prob", 
                    "discriminator_fake_as_real_prob"),
            weight=1.0
        ), # this is the discriminator specific adveserial loss wrapped in LossItem
        LossItem(
            module=gradient_penalty_loss_fn,
            args=("real_target_input_stack", 
                    "fake_target_input_stack",
                    "discriminator"),
            weight=1.0,
            compute_at=('train',) # only compute during training
        ) # this is a special gradient penalty loss specific to wGAN training, requiring the discriminator itself as input
    ]
)

```