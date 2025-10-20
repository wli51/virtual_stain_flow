"""
wgan_losses.py

Wasserstein GAN losses + gradient penalty loss
"""

from typing import Optional

import torch

from .AbstractLoss import AbstractLoss


def _aggr(
        tensor: torch.Tensor,
        dim: int = 3
    ) -> torch.Tensor:
    if tensor.dim() >= dim:
        return torch.mean(
            tensor,
            tuple(range(2, tensor.dim()))
        )        
    else:
        return tensor

class WassersteinDiscriminatorLoss(AbstractLoss):
    """
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def forward(
        p_real_as_real: torch.Tensor,
        p_fake_as_real: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Wasserstein discriminator loss. 
        The Wasserstein loss is defined as the difference between the
            expected probability assigned to real samples and the expected
            probability assigned to fake samples by the discriminator.
        
        :param p_real_as_real: The discriminator's output probabilities for 
            real samples (predicted probability of real samples being real).
        :param p_fake_as_real: The discriminator's output probabilities for
            fake samples (predicted probability of fake samples being real).
        :return: The computed Wasserstein discriminator loss.
        """
        
        return (
            _aggr(p_fake_as_real) - _aggr(p_real_as_real)
        ).mean()

class GradientPenaltyLoss(AbstractLoss):
    """
    Gradient Penalty Loss for Generative Adversarial Networks (GANs).
    This loss function is used to enforce a Lipschitz constraint on the 
        discriminator by penalizing the gradient norm of the discriminator's 
        output with respect to the small perturbations of the input data 
        (interpolated between real and fake data). This helps stabilize the
        training of GANs, particularly in Wasserstein GANs with Gradient Penalty
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def forward(
        input: torch.Tensor,
        target: torch.Tensor,
        fake: torch.Tensor,
        discriminator: torch.nn.Module,
        **kwargs: Optional[dict]
    ) -> torch.Tensor:
        """
        Computes the gradient penalty loss for WGAN-GP.

        :param input: The input tensor to the generator (e.g., low-res image).
        :param target: The real target tensor (e.g., high-res image).
        :param fake: The generated (fake) tensor from the generator.
        :param discriminator: The discriminator module
        :param kwargs: Additional keyword arguments (not used here).
        :return: The computed gradient penalty loss.
        """

        target_input_stack = torch.stack([input, target], dim=1)
        fake_input_stack = torch.stack([input, fake], dim=1)

        device = target_input_stack.device
        dtype = target_input_stack.dtype

        batch_size = target_input_stack.size(0)

        eta = torch.rand(
            # this allows the broadcasting of the same random eta value
            # across all channels, height and width, while each batch sample
            # still gets a (random) different eta.
            batch_size, 1, 1, 1, 
            device=device,
            dtype=dtype,
        ).expand_as(target_input_stack)

        one = torch.as_tensor(
            1., device=device, dtype=dtype)
        interpolated = (
            eta * target_input_stack + (one - eta) * fake_input_stack
        ).requires_grad_(True)
        prob_interpolated = discriminator(interpolated)

        grad_out = torch.ones_like(prob_interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # L2 norm of gradients for each batch sample
        return (
            torch.linalg.vector_norm(
                gradients.flatten(1),
                ord=2,
                dim=1
            ) - 1.0
        ).pow(2).mean()

class AdveserialGeneratorLoss(AbstractLoss):
    """
    Adversarial loss for the generator in a GAN setup.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def forward(
        self,
        p_fake_as_real: torch.Tensor
    ):
        """
        Computes the adversarial loss to train the generator, which is as
            simple as the negative of the discriminator's probabilities for
            predicting fake, generated samples as real.

        :param p_fake_as_real: The discriminator's output probabilities for
            fake samples (predicted probability of fake samples being real).
        :return: The computed adversarial generator loss.
        """
        
        return -_aggr(p_fake_as_real).mean()
