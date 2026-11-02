"""Binary Canny masks for depth regularization."""
import numpy as np
from PIL import Image, ImageFilter


def canny(rgb):
    """Return a binary mask of the detected edge pixels."""
    gray = np.asarray(Image.fromarray(rgb).convert("L").filter(ImageFilter.GaussianBlur(1.2)), dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(gray, 1, mode="edge"), (3,3))
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gx = (windows * sobel).sum((-2,-1))
    gy = (windows * sobel.T).sum((-2,-1))
    mag = np.hypot(gx, gy)
    direction = np.floor((np.rad2deg(np.arctan2(gy,gx)) + 22.5)/45).astype(int) % 4
    thin = np.zeros_like(mag)
    for k, shift in enumerate([(0,1),(1,1),(1,0),(1,-1)]):
        keep = ((direction == k) & (mag >= np.roll(mag, shift, (0,1))) &
                (mag > np.roll(mag, tuple(-i for i in shift), (0,1))))
        thin[keep] = mag[keep]
    thin[[0,-1],:] = thin[:,[0,-1]] = 0
    edges = thin >= 60
    while True:
        grown = (thin >= 25) & dilate(edges)
        if np.array_equal(edges, grown):
            return edges
        edges = grown


def dilate(mask):
    return np.asarray(Image.fromarray(mask.astype('uint8')).filter(ImageFilter.MaxFilter(3))) > 0
