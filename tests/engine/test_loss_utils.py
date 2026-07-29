import torch

from virtual_stain_flow.engine.loss_utils import _scalar_from_ctx


def test_scalar_from_ctx_uses_module_parameter_device_and_dtype():
	module = torch.nn.Linear(1, 1, dtype=torch.float64)

	scalar = _scalar_from_ctx(2.5, {"module": module})

	parameter = next(module.parameters())
	assert scalar.item() == 2.5
	assert scalar.device == parameter.device
	assert scalar.dtype == parameter.dtype
