from torch import Tensor
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from .AbstractLoss import AbstractLoss

class NegativeMultScaleSSIM(AbstractLoss):
    """
    This class implements the negative multi-scale structural similarity index measure (MS-SSIM) loss function.
    It is used to evaluate the similarity between two images, where a lower value indicates higher similarity.
    The loss is computed as the negative of the MS-SSIM score.
    """
    def __init__(self, _metric_name):
        super().__init__(_metric_name)
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11, sigma=1.5)

    def forward(self, pred: Tensor, target: Tensor):
        """
        Computes the negative MS-SSIM loss.
        """
        
        return -self.ms_ssim(pred, target)
