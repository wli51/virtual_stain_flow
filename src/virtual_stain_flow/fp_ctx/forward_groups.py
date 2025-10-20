"""
forward_groups.py

ForwardGroup protocol and implementations for different model architectures.
"""
from typing import Protocol, Set, runtime_checkable

import torch

from .ctx import Context

@runtime_checkable
class ForwardGroup(Protocol):
    """Protocol for defining a group of forward operations in a model."""

    required_inputs: Set[str]
    produce: Set[str]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __call__(
        self,
        train: bool,
        **inputs: torch.Tensor
    ) -> Context: ...


class GeneratorForwardGroup(ForwardGroup):
    """Forward group for simple single generator training."""

    required_inputs = {"inputs", "targets"}
    produce = {"preds", "model"}

    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def __call__(
        self,
        train: bool,
        **inputs: torch.Tensor
    ) -> Context:

        ctx = Context(
            **{
                key: val.to(self.device) for key, val in inputs.items()
            },
            **{"model": self.model}
        )
        ctx.require(self.required_inputs)
            
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            self.model.to(self.device)
            preds = self.model(x=ctx["inputs"].to(self.device))

        return ctx.add(preds=preds)


# class GANGeneratorForwardGroup(ForwardGroup):

#     required_inputs = {"inputs", "targets"}
#     produce = {
#         "pred",
#         "fake_stack",
#         "p_fake_as_real",
#         "generator",
#     }

#     def __init__(
#         self,
#         generator: torch.nn.Module,
#         discriminator: torch.nn.Module,
#     ):
#         self.generator = generator
#         self.discriminator = discriminator
        
#     def __call__(
#         self,
#         train: bool,
#         **inputs: torch.Tensor
#     ):
#         ctx = Context(
#             **inputs,
#             **{
#                 "generator": self.generator,
#                 "discriminator": self.discriminator,
#             }
#         )
#         ctx.require(self.required_inputs)
            
#         if train:
#             self.generator.train()
#         else:
#             self.generator.eval()
#         self.discriminator.eval()
#         self.discriminator.requires_grad_(False)

#         with torch.set_grad_enabled(train):
#             preds = self.generator(x=ctx["inputs"])
#             fake_stack = torch.stack([ctx["inputs"], preds], dim=1)
#             p_fake_as_real = self.discriminator(fake_stack)

#         return ctx.add({
#             "pred": preds,
#             "fake_stack": fake_stack,
#             "p_fake_as_real": p_fake_as_real,
#         })
