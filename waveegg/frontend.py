"""Differentiable reproduction of the archived 2025 inference frontend."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import butter, firwin
from torchaudio.functional import lfilter


class Frontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        sr = config['sample_rate']
        b, a = butter(config['audio_highpass_order'], config['audio_highpass_hz'],
                      btype='highpass', fs=sr)
        self.register_buffer('audio_b', torch.tensor(b, dtype=torch.float64))
        self.register_buffer('audio_a', torch.tensor(a, dtype=torch.float64))
        for name, cutoff, pass_zero in (
            ('egg_high', config['egg_highpass_hz'], False),
            ('egg_low', config['egg_lowpass_hz'], True),
        ):
            coeffs = firwin(config['egg_fir_taps'], cutoff, fs=sr,
                           window='hamming', pass_zero=pass_zero)
            self.register_buffer(name, torch.tensor(coeffs, dtype=torch.float64))

    def audio(self, waveform):
        # Double precision avoids cancellation in the low-cutoff IIR filter.
        # Only fixed coefficient design uses SciPy; waveform processing is Torch.
        return lfilter(waveform.double(), self.audio_a, self.audio_b, clamp=False)

    @staticmethod
    def _fir_with_initial_state(x, b):
        n = b.numel() - 1
        filtered = F.conv1d(F.pad(x[:, None, :], (n, 0)),
                            b.flip(0)[None, None, :])[:, 0, :]
        # scipy.signal.lfilter_zi(b, [1]) for an FIR: reverse cumulative sums.
        zi = b[1:].flip(0).cumsum(0).flip(0)
        initial = x[:, :1] * zi[None, :]
        return filtered + F.pad(initial, (0, x.shape[-1] - n))

    @classmethod
    def _filtfilt(cls, x, b):
        # Match SciPy's default odd extension, padlen=3*len(b).
        edge = 3 * b.numel()
        if x.shape[-1] <= edge:
            raise ValueError(f'Postfilter requires reconstructed length > {edge}; '
                             'use longer audio or explicitly set postfilter=False.')
        left = 2 * x[:, :1] - x[:, 1:edge + 1].flip(-1)
        right = 2 * x[:, -1:] - x[:, -edge - 1:-1].flip(-1)
        extended = torch.cat((left, x, right), dim=-1)
        forward = cls._fir_with_initial_state(extended, b)
        backward = cls._fir_with_initial_state(forward.flip(-1), b).flip(-1)
        return backward[:, edge:-edge]

    def egg(self, waveform):
        return self._filtfilt(self._filtfilt(waveform.double(), self.egg_high),
                             self.egg_low)
