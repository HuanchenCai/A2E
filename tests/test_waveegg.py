import numpy as np
import pytest
import torch
from scipy.signal import butter, lfilter, firwin, filtfilt
from waveegg import WaveEGG, EGGLoss
from utils.model_utils import WaveNet_a2e as OriginalModel
from pathlib import Path

# Small frame workloads are faster and reproducible with bounded CPU threads.
torch.set_num_threads(2)


def signal(length=3200):
    t = np.arange(length) / 16000
    rng = np.random.default_rng(17)
    return (0.3 * np.sin(2*np.pi*173*t) + 0.08 * np.sin(2*np.pi*346*t)
            + 0.002*rng.standard_normal(length)).astype(np.float32)


def original_reference(audio):
    b, a = butter(2, 30, btype='high', fs=16000)
    x = lfilter(b, a, audio)
    frames = np.array([x[i:i+192] for i in range(0, len(x)-191, 32)], dtype=np.float32)
    model = OriginalModel(1, 128, 7, 0.3).eval()
    path = Path(__file__).parents[1] / 'waveegg/weights/trial22.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()})
    with torch.no_grad():
        y = model(torch.from_numpy(frames[:, None, :])).squeeze(1).numpy()
    out = np.zeros((len(y)-1)*32+192)
    den = np.zeros_like(out)
    window = np.hanning(192)
    for i, frame in enumerate(y):
        out[i*32:i*32+192] += frame * window
        den[i*32:i*32+192] += window
    out /= np.where(den == 0, 1, den)
    for cutoff, pass_zero in [(80, False), (4000, True)]:
        b = firwin(1025, cutoff, fs=16000, window='hamming', pass_zero=pass_zero)
        out = filtfilt(b, [1.0], out)
    return x, out


def test_real_checkpoint_matches_original_scipy_pipeline():
    audio = signal()
    x_ref, reference = original_reference(audio)
    extractor = WaveEGG(frame_batch_size=23)
    with torch.no_grad():
        x = extractor.frontend.audio(torch.from_numpy(audio)[None]).numpy()[0]
        result = extractor(torch.from_numpy(audio)[None]).numpy()[0]
    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result, reference, rtol=2e-4, atol=2e-5)
    print('maximum absolute end-to-end difference:', np.max(np.abs(result-reference)))


def test_loss_backpropagates_to_decoder_but_keeps_extractor_frozen():
    extractor = WaveEGG(frame_batch_size=32)
    criterion = EGGLoss(extractor, correlation_weight=0.1)
    decoder = torch.nn.Conv1d(1, 1, 3, padding=1)
    torch.nn.init.constant_(decoder.weight, 0.2)
    torch.nn.init.zeros_(decoder.bias)
    parent = torch.nn.ModuleDict({'decoder': decoder, 'loss': criterion}).train()
    assert all(not m.training for m in extractor.modules())
    before = {k:v.clone() for k,v in extractor.named_buffers()}
    target = torch.from_numpy(signal())[None].requires_grad_(True)
    prediction = parent['decoder'](target.detach()[:,None])[:,0]
    prediction.retain_grad()
    loss = criterion(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert decoder.weight.grad is not None and decoder.weight.grad.abs().sum() > 0
    assert target.grad is None
    assert all(p.grad is None and not p.requires_grad for p in extractor.parameters())
    assert all(torch.equal(before[k],v) for k,v in extractor.named_buffers())


def test_batch_chunk_and_tail_behavior():
    extractor = WaveEGG(frame_batch_size=7)
    audio = torch.from_numpy(signal(3231))[None]
    with torch.no_grad():
        a = extractor(audio)
        extractor.frame_batch_size = 256
        b = extractor(torch.cat((audio,audio)))[0:1]
    assert a.shape == (1, 3200)
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    with pytest.raises(ValueError, match='3075'):
        extractor(torch.zeros(1,192))
    with pytest.raises(ValueError, match='192'):
        extractor(torch.zeros(1,191))
    with pytest.raises(ValueError, match='float32'):
        extractor(audio.double())


def test_silence_and_explicit_short_clip_mode():
    extractor = WaveEGG(postfilter=False)
    x = torch.zeros(1,192,requires_grad=True)
    loss = EGGLoss(extractor,correlation_weight=0.1)(x,torch.zeros_like(x))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert loss.item() == pytest.approx(0,abs=1e-6)
