from typing import Optional

import torch

from .AbstractLoss import AbstractLoss

"""
Wasserstein Discriminator Loss for Generative Adversarial Networks (GANs).

This loss function is for training the discriminator to return a scalar 
expected probability score for a target-input stack being real. 
We want the discriminator to output a high expected probability score for a 
ground_truth-input stack and a low expected probability score for a 
generated-input stack.
"""
class WassersteinDiscriminatorLoss(AbstractLoss):

    def __init__(
        self,
        metric_name: str = "WassersteinDiscriminatorLoss",
    ):
        super().__init__(metric_name)

    def forward(
        self, 
        expected_real_as_real_prob: torch.Tensor,
        expected_fake_as_real_prob: torch.Tensor,
        **kwargs: Optional[dict]
    ):
        """
        Computes the Wasserstein Discriminator Loss

        :param expected_real_as_real_prob: Discriminator's output given a 
            batch of ground_truth-input stack.
        :param expected_fake_as_real_prob: Discriminator's output given a
            batch of generated-input stack.
        :param kwargs: Additional keyword arguments, not used in this loss.
        """
        
        # Need to handle the case where the discriminator outputs more than a B, scalar
        if expected_real_as_real_prob.dim() >= 3:
            expected_real_as_real_prob = torch.mean(
                expected_real_as_real_prob, 
                tuple(range(2, expected_real_as_real_prob.dim()))
            )
        
        if expected_fake_as_real_prob.dim() >= 3:
            expected_fake_as_real_prob = torch.mean(
                expected_fake_as_real_prob, 
                tuple(range(2, expected_fake_as_real_prob.dim()))
            )

        return (
                expected_fake_as_real_prob - \
                expected_real_as_real_prob
            ).mean()
    
"""
Gradient Penalty Loss for Generative Adversarial Networks (GANs).
This loss function is used to enforce a Lipschitz constraint on the discriminator
by penalizing the gradient norm of the discriminator's output with respect to 
the small perturbations of the input data (interpolated between real and fake data).
"""
class GradientPenaltyLoss(AbstractLoss):
    def __init__(
        self,
        metric_name: str = "GradientPenaltyLoss",
    ):
        super().__init__(metric_name)

    def forward(
        self, 
        target: torch.Tensor,
        predicted: torch.Tensor,
        discriminator: torch.nn.Module,
        **kwargs: Optional[dict]
    ):
        """
        Computes the Gradient Penalty Loss for the discriminator.

        :param target: The tensor containing the ground truth image,
            should be of shape (B, C, H, W).
        :param predicted: The tensor containing the model generated image,
            should be of shape (B, C, H, W).
        :param discriminator: The discriminator model.
        :param kwargs: Additional keyword arguments, not used in this loss.
        """
        
        device = self.trainer.device
        batch_size = target.size(0)

        eta = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(target)
        interpolated = (eta * target + (1 - eta) * predicted).requires_grad_(True)
        prob_interpolated = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
"""
Adversarial Generator Loss for Generative Adversarial Networks (GANs).
This loss function is for training the generator to produce images that
the discriminator classifies as real.
The generator aims to maximize the probability that the discriminator
classifies the generated images as real.
"""
class AdveserialGeneratorLoss(AbstractLoss):
    def __init__(
        metric_name: str = "AdversarialGeneratorLoss",
    ):
        super().__init__(metric_name)

    def forward(
        self, 
        expected_fake_as_real_prob: torch.Tensor,
        **kwargs: Optional[dict]
    ):
        """
        Computes the Adversarial Generator Loss

        :param expected_fake_as_real_prob: Discriminator's output given a 
            batch of generated-input stack.
        :param kwargs: Additional keyword arguments, not used in this loss.
        """
        
        # Need to handle the case where the discriminator outputs more than a B, scalar
        if expected_fake_as_real_prob.dim() >= 3:
            expected_fake_as_real_prob = torch.mean(
                expected_fake_as_real_prob, 
                tuple(range(2, expected_fake_as_real_prob.dim()))
            )

        return -expected_fake_as_real_prob.mean()