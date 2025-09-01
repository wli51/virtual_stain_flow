from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
import torch.nn as nn

class AbstractMetrics(nn.Module, ABC):
    """Abstract class for metrics"""

    def __init__(self, _metric_name: str):
        
        super(AbstractMetrics, self).__init__()

        self.__metric_name = _metric_name
        
        self.__train_metric_values = []
        self.__val_metric_values = []

    @property
    def metric_name(self):
        """Defines the mertic name returned by the class."""
        return self.__metric_name
    
    @property
    def train_metric_values(self):
        """Returns the training metric values."""
        return self.__train_metric_values
    
    @property
    def val_metric_values(self):
        """Returns the validation metric values."""
        return self.__val_metric_values

    @abstractmethod
    def forward(self, 
                _generated_outputs: torch.tensor, 
                _targets: torch.tensor
                ) -> torch.tensor:
        """Computes the metric given information about the data."""
        pass
    
    @torch.inference_mode()
    def update(self, 
               _generated_outputs: torch.tensor, 
               _targets: torch.tensor, 
               validation: bool=False
               ) -> None:
        """Updates the metric with the new data."""

        val = self.forward(_generated_outputs, _targets)

        if val.numel() > 1:
            val = val.float().mean()
        v = float(val.detach().item())

        if validation:
            self.__val_metric_values.append(v)
        else:
            self.__train_metric_values.append(v)

    def reset(self):
        """Resets the metric."""
        self.__train_metric_values.clear()
        self.__val_metric_values.clear()

    def compute(self, **kwargs):
        """
        Calls the aggregate_metrics method to compute the metric value for now
        In future may be used for more complex computations
        """
        return self.aggregate_metrics(**kwargs)

    def aggregate_metrics(
        self, 
        aggregation: Optional[Literal['mean', 'sum']]='mean'
    ) -> tuple[Optional[torch.tensor], Optional[torch.tensor]]:
        """
        Aggregates the metric value over batches. Returns the training and
            validation metric values.

        :param aggregation: The aggregation method to use, by default 'mean'
        :type aggregation: Optional[str]
        :return: The aggregated metric value for training and validation
        """

        tr_list, va_list = self.__train_metric_values, self.__val_metric_values

        if aggregation == 'mean':
            tr = torch.tensor(sum(tr_list)/len(tr_list)) if tr_list else None
            va = torch.tensor(sum(va_list)/len(va_list)) if va_list else None
            return tr, va
        
        elif aggregation == 'sum':
            tr = torch.tensor(sum(tr_list)) if tr_list else None
            va = torch.tensor(sum(va_list)) if va_list else None
            return tr, va
        
        elif aggregation is None:
            tr = torch.tensor(tr_list, dtype=torch.float32) if tr_list else None
            va = torch.tensor(va_list, dtype=torch.float32) if va_list else None
            return tr, va
        
        else:
            raise ValueError(
                f"Aggregation method {aggregation} is not supported.")