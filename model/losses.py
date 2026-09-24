import torch
from torch.nn import CrossEntropyLoss


class OAMaskedCrossEntropyLoss(CrossEntropyLoss):
    """Order-agnostic cross-entropy evaluated only at masked positions."""

    def __init__(self, weight=None, reduction="none", reweight=True):
        self.reweight = reweight
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, pred, tgt, mask, timesteps, input_mask):
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(-1)
            input_mask = input_mask.unsqueeze(-1)

        mask = mask.bool()
        input_mask = input_mask.bool()
        masked_tokens = mask.sum()
        nonpadding_tokens = input_mask.sum(dim=1)
        masked_pred = torch.masked_select(pred, mask).view(masked_tokens, -1)
        masked_target = torch.masked_select(tgt, mask.squeeze())
        token_loss = super().forward(masked_pred, masked_target)
        nll = token_loss.sum()

        if not self.reweight:
            return nll, nll.to(torch.float64)

        weights = (1.0 / timesteps).repeat_interleave(timesteps)
        sequence_lengths = nonpadding_tokens.repeat_interleave(timesteps)
        loss = (sequence_lengths * weights * token_loss).sum()
        return loss, nll.to(torch.float64)
