"""Frozen WaveEGG extractor retaining gradients with respect to input audio."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .frontend import Frontend
from .model import WaveNet_a2e


class WaveEGG(nn.Module):
    """Input [batch, time] float32, 16 kHz. Output [batch, covered_time].

    Default settings reproduce the archived 2025 Trial22 test frontend.
    Incomplete tail frames are omitted; use covered_length() to align targets.
    """
    def __init__(self, *, postfilter=None, frame_batch_size=256):
        super().__init__()
        if not isinstance(frame_batch_size, int) or frame_batch_size < 1:
            raise ValueError('frame_batch_size must be a positive integer')
        root = Path(__file__).parent
        self.config = json.loads((root / 'config.json').read_text(encoding='utf-8-sig'))
        self.frame_batch_size = frame_batch_size
        self.postfilter = self.config['postfilter'] if postfilter is None else bool(postfilter)
        self.model = WaveNet_a2e(**self.config['architecture'])
        path = root / 'weights' / 'trial22.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.config['checkpoint_sha256']:
            raise ValueError('Trial22 checkpoint checksum mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.frontend = Frontend(self.config)
        self.register_buffer('window', torch.hann_window(self.config['frame_length'],
                                                        periodic=False, dtype=torch.float64))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent decoder.train() must not change pretrained BN/Dropout behavior.
        return super().train(False)

    def covered_length(self, length):
        frame, hop = self.config['frame_length'], self.config['inference_hop']
        if length < frame:
            raise ValueError(f'Input needs at least {frame} samples')
        return ((length - frame) // hop) * hop + frame

    def overlap_add(self, frames):
        # frames: [batch, frames, samples]. Match np.hanning + float64 sums.
        batch, count, width = frames.shape
        hop = self.config['inference_hop']
        length = (count - 1) * hop + width
        columns = (frames.double() * self.window).transpose(1, 2)
        result = F.fold(columns, (1, length), (1, width), stride=(1, hop))[:, 0, 0, :]
        weights = self.window[None, :, None].expand(1, width, count)
        denominator = F.fold(weights, (1, length), (1, width), stride=(1, hop))[0, 0, 0]
        return result / denominator.masked_fill(denominator == 0, 1)

    def forward(self, audio):
        if audio.ndim != 2 or audio.dtype != torch.float32:
            raise ValueError('Expected float32 audio [batch, time], sampled at 16000 Hz')
        if audio.shape[0] == 0 or not torch.isfinite(audio).all():
            raise ValueError('Audio must contain a nonempty batch of finite values')
        if audio.device != self.window.device:
            raise ValueError('Move WaveEGG and audio to the same device')
        length = self.covered_length(audio.shape[-1])
        if self.postfilter and length <= 3 * self.config['egg_fir_taps']:
            raise ValueError('Default postfilter requires >3075 covered samples '
                             '(about 0.2 seconds); set postfilter=False explicitly for shorter audio.')
        x = self.frontend.audio(audio).float()
        frames = x.unfold(-1, self.config['frame_length'], self.config['inference_hop'])
        batch, count, width = frames.shape
        flat = frames.reshape(-1, 1, width)
        # Disable ambient mixed precision: this baseline is validated in float32.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            predictions = torch.cat([self.model(chunk) for chunk in flat.split(self.frame_batch_size)])
            result = self.overlap_add(predictions.reshape(batch, count, width))
            if self.postfilter:
                result = self.frontend.egg(result)
        return result.float()
