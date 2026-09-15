# A2E / WaveEGG

Audio-to-EGG waveform prediction, with an archived **Trial22 checkpoint** and a
frozen, differentiable EGG-space auxiliary loss for audio-generating models.

## Quick start

```bash
git clone https://github.com/HuanchenCai/A2E.git
cd A2E
python -m venv .venv
# Activate the environment using your platform's usual command.
python -m pip install -e ".[test]"
python examples/auxiliary_loss.py
python -m pytest tests -q
```

For a CPU-only environment, install PyTorch and torchaudio from the official CPU
wheel index before the editable install:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

The original root `requirements.txt` belongs to the historical notebooks. The new
`waveegg` package declares its own dependencies in `pyproject.toml`.
`requirements-waveegg-tested-cpu.txt` records the exact tested CPU environment.
CUDA has not been validated for this release; choose compatible PyTorch/audio
builds for your platform and run the tests before integration.

## Using WaveEGG as an auxiliary loss

```python
from waveegg import EGGLoss

criterion = EGGLoss().to(device)
# Float32 tensors [batch, samples], mono, 16000 Hz, equal lengths and aligned.
predicted_audio = decoder(neural_input)
loss_egg = criterion(predicted_audio, reference_audio)
loss = loss_audio + lambda_egg * loss_egg
loss.backward()
```

`EGGLoss` defaults to MSE. Optional `correlation_weight=0.1` adds 1-Pearson
correlation with low-variance protection; this coefficient is an example, not a
validated setting for your decoder. Keep your primary audio objective and tune
the auxiliary weight experimentally.

The extractor is frozen and stays in evaluation mode even when a parent module
receives `.train()`. **Do not wrap the predicted-audio branch in `no_grad()` or
convert its waveform to NumPy.** Reference extraction is detached inside EGGLoss.
The upstream audio decoder/vocoder must itself support differentiation.

Without synchronized measured EGG, this loss compares two audio-derived
**pseudo-EGG** signals. It is not supervision from a physiological measurement,
and improvement to the collaborator's decoder is not established by these tests.

For direct extraction or reference caching:

```python
import torch
from waveegg import WaveEGG

extractor = WaveEGG().to(device)
with torch.no_grad():
    target_egg = extractor(reference_audio)
predicted_egg = extractor(predicted_audio)  # gradients retained
# EGGLoss excludes the first/last reconstructed samples (Hann endpoints).
loss_egg = (predicted_egg[:, 1:-1] - target_egg[:, 1:-1]).square().mean()
```

Cache reference features only when the checkpoint, frontend, augmentation,
cropping and frame boundaries are identical. Process variable-length recordings
individually or in equal-length groups: padded samples are not automatically
masked by this API. Resample to 16 kHz before calling; use a differentiable
resampler if the generated audio needs resampling.

## Exact frontend and input contract

The default frontend follows the archived February 2025 **Trial22 test notebook**:

1. Float32 mono waveform `[batch, time]`, sampled at 16 kHz. No automatic
   resampling, clipping, peak scaling or per-batch normalization.
2. 30 Hz, second-order Butterworth high-pass, causal `lfilter`, zero initial state.
   Filter arithmetic uses float64; the network receives float32 frames.
3. 192-sample (12 ms) rectangular input frames, **32-sample inference hop**.
4. WaveNet_a2e: 7 layers, 128 gated channels, dropout 0.3, frozen BatchNorm,
   self-attention and residual/skip paths. The existing attention multiplication
   and first-layer residual broadcasting are preserved exactly.
5. Symmetric Hann overlap-add, divided by accumulated window weights.
6. Output EGG filtering: 1025-tap Hamming FIR high-pass at 80 Hz, then low-pass at
   4000 Hz. Each uses forward/backward filtering with SciPy-compatible default
   odd padding and initial conditions. Coefficient design uses SciPy; waveform
   filtering and reconstruction use differentiable Torch operations.
7. Float32 output. No extra output softmax, polarity flip or 44.1 kHz conversion.

The two fixed normalization divisors in the older GitHub test notebook are
intentionally **not applied**, matching the archived 2025 notebook. This frontend
choice is not proof of the exact preprocessing used to train Trial22.

Only complete frames are used. For input length T, output length is
`((T - 192) // 32) * 32 + 192`. Incomplete tails are omitted. No arbitrary
zero-padding is added. `extractor.covered_length(T)` returns this length.

The default output filter requires **more than 3075 reconstructed samples**;
3200 samples (0.2 seconds) is a convenient minimum. For explicit short-clip
experiments, `WaveEGG(postfilter=False)` accepts at least 192 samples, but produces
a different feature space. Use the same setting on both loss branches. The
unfiltered OLA endpoints are zero; EGGLoss omits the first and last reconstructed
samples, including when postfiltering is enabled. Boundary filter effects remain
part of the defined frontend.

Use `.to(device)` without casting the extractor to half precision. The validated
network path is float32 and filtering is float64. `frame_batch_size` changes only
how complete frames are batched, not their grid; autograd still retains the graph
for all frames. Long utterances can require substantial training memory.

## Checkpoint and provenance

The actual weight file is committed at
[`waveegg/weights/trial22.pt`](waveegg/weights/trial22.pt) (3,476,210 bytes).
It is a state_dict containing parameters and BatchNorm buffers, with historical
`module.` prefixes. Loading verifies SHA-256, removes only the leading prefix,
and uses `weights_only=True`, CPU mapping and strict key/shape checking.

- Archive identifier: `Trial22_stride4_length12ms_dl7_lr0.001_gamma0.95_ALL`.
- SHA-256: `de88b1c9787278f7463877dc282403b62f08defdf5ac610f274a944bf1129b96`.
- 856,161 trainable parameters before freezing; 107 state_dict entries.
- Training checkpoint records: initial LR 0.001, exponential gamma 0.95,
  weight_decay 0. The archive includes checkpoint numbering up to 125.
- The original public main at `cff32b34558457516289af63b50952ec974a2842`
  referenced **Trial21**, not this Trial22 checkpoint.
- Trial22's name suggests training hop 4, while the 2025 inference notebook uses
  hop 32. Training normalization and the final dataset split are not fully
  established by this bundle.

**This is an identified archival checkpoint, not a verified reproduction of all
final JASA paper results.** The exact link between Trial22 and the February 2025
paper result tables is still unconfirmed. The network's executable source is the
architecture authority; do not add layers from schematic diagrams or reinterpret
it as a sample-by-sample autoregressive generator.

Paper: [A WaveNet-Based model for predicting the electroglottographic signal
from the acoustical signal](https://doi.org/10.1121/10.0036514).

## Validation

Validated on Windows CPU, Python 3.12.10, PyTorch 2.14.0+cpu,
torchaudio 2.11.0+cpu, SciPy 1.18.1, NumPy 2.5.3:

- Strict loading of the real Trial22 state_dict.
- End-to-end comparison with the original model plus independent NumPy/SciPy
  preprocessing and overlap-add on a deterministic synthetic voiced signal.
  Observed maximum absolute difference: **7.43e-9**.
- A real Conv1d decoder receives finite, nonzero gradients through EGGLoss.
- Reference input and frozen extractor parameters receive no gradients;
  BatchNorm buffers remain unchanged under parent `.train()`.
- Batch/chunk consistency, silence, incomplete tails and short-clip validation.

Run `python -m pytest tests -q` in the checkout. This validates implementation
parity and gradient flow, not speech quality or physiological accuracy on new data.

## Files to use

- `waveegg/model.py`: matching model definitions.
- `waveegg/frontend.py`: differentiable filters.
- `waveegg/inference.py`: checkpoint loading and frozen extraction.
- `waveegg/losses.py`: auxiliary loss.
- `waveegg/config.json` and `waveegg/weights/trial22.pt`: explicit settings/weights.
- `examples/auxiliary_loss.py` and `tests/test_waveegg.py`: integration and checks.

The historical `train.py`, notebooks, utility backups and plotting code are kept
for context. They mix several architectures and experimental settings and are
not the entry point for this package. No training recordings were added in this
release.
