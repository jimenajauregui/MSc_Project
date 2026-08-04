import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDecouplingLoss(nn.Module):
    """
    Spectral Decoupling Loss: BCE + L2 penalty on raw unnormalized logits.
    Decouples feature representation learning from logit norm growth under distribution shift.
    Ref: Farquhar et al. (2020), Chalkidis & Søgaard (2022)
    """
    def __init__(self, lambda_sd: float = 0.01, reduction: str = 'mean'):
        super().__init__()
        self.lambda_sd = lambda_sd
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)
        logit_norm_penalty = 0.5 * self.lambda_sd * torch.mean(logits ** 2)
        return bce_loss + logit_norm_penalty