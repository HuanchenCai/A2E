"""EGG-space auxiliary loss, with reference features detached."""
import torch
from torch import nn
from .inference import WaveEGG


class EGGLoss(nn.Module):
    def __init__(self, extractor=None, *, correlation_weight=0.0):
        super().__init__()
        if correlation_weight < 0:
            raise ValueError('correlation_weight must be nonnegative')
        self.extractor = WaveEGG() if extractor is None else extractor
        self.correlation_weight = correlation_weight

    def forward(self, predicted_audio, reference_audio):
        if predicted_audio.shape != reference_audio.shape:
            raise ValueError('Predicted and reference audio must be aligned and have equal shapes')
        prediction = self.extractor(predicted_audio)
        with torch.no_grad():
            target = self.extractor(reference_audio)
        # Exclude the two zero-weight Hann endpoints; incomplete tails are omitted.
        prediction, target = prediction[:, 1:-1], target[:, 1:-1]
        mse = (prediction - target).square().mean()
        if self.correlation_weight == 0:
            return mse
        p = prediction - prediction.mean(-1, keepdim=True)
        t = target - target.mean(-1, keepdim=True)
        pvar, tvar = p.square().mean(-1), t.square().mean(-1)
        denominator = (pvar.clamp_min(1e-8) * tvar.clamp_min(1e-8)).sqrt()
        correlation = (p * t).mean(-1) / denominator
        # A constant reference carries no correlation target; MSE still applies.
        valid = tvar > 1e-8
        corr_loss = ((1 - correlation.clamp(-1, 1)) * valid).sum() / valid.sum().clamp_min(1)
        return mse + self.correlation_weight * corr_loss
