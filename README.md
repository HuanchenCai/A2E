# A2E / WaveEGG

**Predicting electroglottographic (EGG) waveforms from the acoustic voice signal.**

We investigate whether microphone recordings can be used to estimate EGG waveforms and characterize phonation. Our model combines a WaveNet-based architecture with self-attention to learn the relationship between acoustic voice signals and EGG.

This repository contains our research code, a pretrained checkpoint, and a PyTorch interface for waveform prediction.

## Paper

Huanchen Cai and Sten Ternström (2025). **A WaveNet-based model for predicting the electroglottographic signal from the acoustic voice signal.** *The Journal of the Acoustical Society of America (JASA)*, **157**(4), 3033–3044.

[Read the paper](https://doi.org/10.1121/10.0036514)

If you use this work in your research, please cite our paper.

## Getting started

```bash
git clone https://github.com/HuanchenCai/A2E.git
cd A2E
python -m pip install -e .
```

```python
import torch
from waveegg import WaveEGG

model = WaveEGG()
audio = torch.zeros(1, 16000)  # Replace with mono audio: float32, 16 kHz
with torch.no_grad():
    predicted_egg = model(audio)
```

The included checkpoint is **Trial22**. The interface accepts batches of 16 kHz mono audio and returns estimated EGG waveforms. Use recordings of at least 0.2 seconds with the default settings.

The package also supports EGG-space auxiliary losses through `EGGLoss`. See the [usage notes](docs/usage.md) for preprocessing, checkpoint provenance, and differentiable use, or run the [example](examples/auxiliary_loss.py).

## Repository

- `waveegg/` — model, pretrained weights, and inference interface.
- `notebooks/` and `utils/` — research experiments and analysis tools.
- `examples/` and `tests/` — usage examples and implementation checks.
