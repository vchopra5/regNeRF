"""Small calculations shared by the three lab notebooks."""

from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = "https://www.cvlibs.net/publications/Niemeyer2022CVPR.pdf"


def solve_depth(observed, strength, weights=None):
    """Minimize sum((depth-observed)^2) + strength*sum(w*diff(depth)^2)."""
    n = len(observed)
    difference = np.diff(np.eye(n), axis=0)
    weights = np.ones(n - 1) if weights is None else np.asarray(weights)
    return np.linalg.solve(np.eye(n) + strength * difference.T @
                           (weights[:, None] * difference), observed)


def solve_depth_l1(observed, strength, weights=None, iterations=100, eps=1e-4):
    """Minimize sum((depth-observed)^2) + strength*sum(w*|diff(depth)|).

    Iteratively reweighted least squares: each pass replaces |x| with x^2 / |x_previous|.
    """
    n = len(observed)
    difference = np.diff(np.eye(n), axis=0)
    weights = np.ones(n - 1) if weights is None else np.asarray(weights)
    depth = np.asarray(observed, dtype=float).copy()
    for _ in range(iterations):
        scale = weights / np.maximum(np.abs(difference @ depth), eps)
        depth = np.linalg.solve(np.eye(n) + 0.5 * strength * difference.T @
                                (scale[:, None] * difference), observed)
    return depth


def sphere_profile():
    """Read one central foreground row. Do not smooth across invalid depth."""
    depth = np.load(ROOT / "toy_sphere/train/r_0_depth.npy")
    row = depth[depth.shape[0] // 2]
    return row[row > 0].astype(float)


def extension_case(case, seed=42):
    """Make two flat surfaces and an independent colour guide."""
    x = np.arange(128)
    truth = np.where(x < 64, 2.0, 3.0)
    colour = np.where(x < 64, 0.2, 0.8)
    if case == "Texture edge":
        colour[24:40] = 0.8
    elif case == "Missing colour edge":
        colour[:] = 0.5
    observed = truth + np.random.default_rng(seed).normal(0, 0.12, len(x))
    # Use reference depth only to define evaluation regions.
    boundary = np.abs(x - 63.5) <= 3.5
    flat = np.abs(x - 63.5) >= 12.5
    return truth, observed, colour, boundary, flat


def rmse(prediction, truth, mask=None):
    error = prediction - truth
    if mask is not None:
        error = error[mask]
    return float(np.sqrt(np.mean(error ** 2)))


def plot(series, title, ylabel="Depth (scene units)"):
    """Return an SVG with shared axes for all curves."""
    colours = ["#18263b", "#a8adb8", "#2878bb", "#d26526"]
    values = np.concatenate([np.asarray(v) for v in series.values()])
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.08, 0.02)
    low, high = low - margin, high + margin
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 350" role="img" aria-label="{escape(title)}">',
           '<rect width="840" height="350" fill="white"/>',
           f'<text x="65" y="25" font-size="18">{escape(title)}</text>',
           '<path d="M65 55V280H810" fill="none" stroke="#526070"/>']
    for fraction in (0, 0.5, 1):
        y = 280 - 225 * fraction
        svg.append(f'<text x="5" y="{y+4}" font-size="12">{low+(high-low)*fraction:.2f}</text>')
    for i, (label, vector) in enumerate(series.items()):
        points = " ".join(f"{65+j*745/(len(vector)-1):.2f},{280-(v-low)*225/(high-low):.2f}"
                          for j, v in enumerate(vector))
        colour = colours[i % len(colours)]
        svg.extend([f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>',
                    f'<text x="{65+i*190}" y="330" fill="{colour}" font-size="13">{escape(label)}</text>'])
    svg.extend([f'<text x="65" y="48" font-size="12">{escape(ylabel)}</text>',
                '<text x="355" y="302" font-size="12">Pixel index (left to right)</text>', '</svg>'])
    return "".join(svg)
