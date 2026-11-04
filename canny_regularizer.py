"""Use the blur and gradient steps of Canny to weight depth differences."""
import torch
import torch.nn.functional as F


# Gaussian radius 4, followed by Sobel radius 1.
HALO = 5
WEIGHT_FLOOR = 0.05


@torch.no_grad()
def canny_depth_weights(rgb, threshold=0.1, floor=WEIGHT_FLOOR):
    """Return x and y weights for RGB patches [N,H,W,3] in [0,1].

    Supply a five-pixel halo. The output covers the inner patch pairs.
    Stop guide gradients so the loss cannot change RGB to reduce its weight.
    """
    if not 0 < threshold < float("inf") or not 0 <= floor <= 1:
        raise ValueError("Use a finite positive threshold and a floor in [0,1].")
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or min(rgb.shape[1:3]) < 12:
        raise ValueError("Use RGB patches with a halo and an inner size of at least 2.")
    gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])).sum(-1).unsqueeze(1)
    axis = torch.arange(-4, 5, device=rgb.device, dtype=rgb.dtype)
    gaussian = torch.exp(-0.5 * (axis / 1.2).square())
    gaussian = gaussian / gaussian.sum()
    # Valid filters remove the halo. No patch padding enters the result.
    blurred = F.conv2d(gray, (gaussian[:, None] * gaussian[None, :])[None, None])
    sobel = rgb.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    gradients = F.conv2d(blurred, torch.stack((sobel, sobel.T)).unsqueeze(1))
    gx, gy = gradients.unbind(1)
    # Average squared gradients at both ends of each depth pair.
    sx = 0.5 * (gx[:, :-1, :-1].square() + gx[:, :-1, 1:].square())
    sy = 0.5 * (gy[:, :-1, :-1].square() + gy[:, 1:, :-1].square())
    return tuple(floor + (1 - floor) * torch.exp(-s / (2 * threshold**2))
                 for s in (sx, sy))


def canny_edge_l1_loss(depth, opacity, smooth_weights, margin=0.1):
    """Penalize the L1 shortfall below a minimum depth change at colour edges.

    Edge confidence is the complement of the default Canny smoothing weight.
    At an exact depth tie, choose the positive direction to start separation.
    This choice supplies a gradient; it does not infer which surface is closer.
    """
    if not 0 < margin < float("inf"):
        raise ValueError("Use a finite positive depth margin.")
    anchor = depth[..., :-1, :-1]
    alpha = opacity[..., :-1, :-1].detach()
    loss = torch.zeros_like(anchor)
    for neighbor, weight in zip((depth[..., :-1, 1:], depth[..., 1:, :-1]), smooth_weights):
        difference = anchor - neighbor
        # Same value as abs(); use a positive subgradient at an exact zero.
        magnitude = torch.where(difference.detach() >= 0, difference, -difference)
        confidence = ((1 - weight.detach()) / (1 - WEIGHT_FLOOR)).clamp(0, 1)
        loss = loss + confidence * F.relu(margin - magnitude)
    return (alpha * loss).mean()
