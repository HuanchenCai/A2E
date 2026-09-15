"""Run from an installed checkout: python examples/auxiliary_loss.py."""
import torch
from waveegg import EGGLoss


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # A small stand-in for a differentiable audio decoder.
    decoder = torch.nn.Conv1d(1, 1, 3, padding=1).to(device)
    egg_loss = EGGLoss().to(device)
    t = torch.arange(3200, device=device) / 16000
    reference = (0.2 * torch.sin(2 * torch.pi * 160 * t))[None]
    predicted = decoder(reference[:, None])[:, 0]
    predicted.retain_grad()
    auxiliary = egg_loss(predicted, reference)
    audio_loss = (predicted - reference).square().mean()
    # Illustrative coefficient; tune for your own decoder and objective.
    loss = audio_loss + 0.01 * auxiliary
    loss.backward()
    assert decoder.weight.grad is not None
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert all(p.grad is None for p in egg_loss.parameters())
    print(f'EGG loss: {auxiliary.item():.6f}; input gradient and decoder gradient verified.')


if __name__ == '__main__':
    main()
