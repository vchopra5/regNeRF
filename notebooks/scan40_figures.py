"""Figures for notebook 2. They use the same 2x canvas, type scale, and ink as the toy figures.

Colours come from a validated palette (dataviz skill, light surface, all checks pass):
one hue for single-series bars, blue/red for better/worse, and blue/orange/aqua for the
camera groups. Text always uses the ink colours, never a data colour.
"""
import numpy as np
from PIL import Image

from notebooks.toy_figures import GRID, INK, LABEL, MUTED, SMALL, TITLE, Board

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
OTHER = "#c9cfd3"


def table3_chart(rows, highlight=("Full RegNeRF", "Without geometry regularization")):
    """Horizontal bars for the paper's ablation. One series, so no legend: the title names it."""
    board = Board(600, 34 + 40 * len(rows) + 40)
    left, right, top = 236, 480, 34
    high = 20.0
    board.text((0, 4), "PSNR on DTU, 3 views (dB)", size=TITLE)
    for tick in range(0, 21, 5):
        x = left + (right - left) * tick / high
        board.line((x, top - 4, x, top + 40 * len(rows)), fill=GRID)
        board.text((x, top + 40 * len(rows) + 4), str(tick), fill=MUTED, size=SMALL, anchor="ma")
    centres = {}
    for i, (name, value) in enumerate(rows):
        y = top + 40 * i + 20
        centres[name] = y
        board.text((left - 10, y), name, fill=INK, size=LABEL, anchor="rm")
        board.bar((left, y - 11, left + (right - left) * value / high, y + 11), fill=BLUE)
        board.text((left + (right - left) * value / high + 6, y), f"{value:.2f}", fill=INK, size=LABEL, anchor="lm")
    # Bracket between the two rows the text compares.
    a, b = (centres[name] for name in highlight)
    values = dict(rows)
    x = left + (right - left) * values[highlight[0]] / high + 48
    board.line([(x - 4, a), (x, a), (x, b), (x - 4, b)], fill=INK, width=1.2)
    board.text((x + 6, (a + b) / 2), f"{values[highlight[0]] - values[highlight[1]]:.2f} dB",
               fill=INK, size=LABEL, anchor="lm")
    return board.image


def camera_map(poses, train_ids, test_ids, highlight):
    """Direction of each camera from the scene centre: left-right angle and height angle."""
    centres = poses[:, :3, 3]
    up = poses[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    mean_dir = centres.mean(0) - up * (centres.mean(0) @ up)
    forward = mean_dir / np.linalg.norm(mean_dir)
    side = np.cross(up, forward)
    d = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    azimuth = np.degrees(np.arctan2(d @ side, d @ forward))
    elevation = np.degrees(np.arcsin(np.clip(d @ up, -1, 1)))

    board = Board(560, 300)
    x0, x1, y0, y1 = 50, 540, 60, 250
    pad = 6
    az_lo, az_hi = azimuth.min() - pad, azimuth.max() + pad
    el_lo, el_hi = elevation.min() - pad, elevation.max() + pad

    def place(a, e):
        return x0 + (a - az_lo) / (az_hi - az_lo) * (x1 - x0), y1 - (e - el_lo) / (el_hi - el_lo) * (y1 - y0)

    board.text((0, 4), "Where the 49 cameras look from (angle around the arch, angle of height)", size=TITLE)
    legend = [("training (3)", BLUE), ("test (25)", ORANGE), (f"view {highlight}", AQUA), ("not used", OTHER)]
    x = 0
    for label, colour in legend:
        if colour == AQUA:
            board.polygon([(x + 6, 30), (x + 12, 36), (x + 6, 42), (x, 36)], fill=colour)
        else:
            board.dot(x + 6, 36, 5, fill=colour, outline="white", width=2)
        board.text((x + 16, 29), label, fill=MUTED, size=SMALL)
        x += 120
    for tick in range(int(np.ceil(az_lo / 15) * 15), int(az_hi) + 1, 15):
        px, _ = place(tick, el_lo)
        board.line((px, y0, px, y1), fill=GRID)
        board.text((px, y1 + 6), f"{tick}°", fill=MUTED, size=SMALL, anchor="ma")
    for tick in range(int(np.ceil(el_lo / 10) * 10), int(el_hi) + 1, 10):
        _, py = place(az_lo, tick)
        board.line((x0, py, x1, py), fill=GRID)
        board.text((x0 - 6, py), f"{tick}°", fill=MUTED, size=SMALL, anchor="rm")
    board.text(((x0 + x1) / 2, y1 + 24), "angle around the arch (left to right)", fill=MUTED, size=SMALL, anchor="ma")
    groups = [(set(range(len(poses))) - set(train_ids) - set(test_ids), OTHER, 4),
              (set(test_ids) - {highlight}, ORANGE, 5), (set(train_ids), BLUE, 6)]
    for ids, colour, radius in groups:
        for i in ids:
            board.dot(*place(azimuth[i], elevation[i]), radius, fill=colour, outline="white", width=2)
    hx, hy = place(azimuth[highlight], elevation[highlight])
    board.polygon([(hx, hy - 8), (hx + 8, hy), (hx, hy + 8), (hx - 8, hy)], fill=AQUA)
    board.text((hx + 11, hy), str(highlight), fill=INK, size=LABEL, anchor="lm")
    for i in train_ids:
        px, py = place(azimuth[i], elevation[i])
        board.text((px, py - 10), str(i), fill=INK, size=SMALL, anchor="md")
    return board.image


def view_grid(photo, rgb_off, rgb_on, depth_off, depth_on, depth_range, width=300):
    """Photo and both renders on top, both depth maps below, one depth scale."""
    height = round(width * rgb_off.height / rgb_off.width)
    board = Board(16 + 3 * (width + 16), 28 + 2 * height + 34)
    columns = [("Real photo", photo, None), ("Smoothing off", rgb_off, depth_off), ("Smoothing on", rgb_on, depth_on)]
    for i, (title, colour, depth) in enumerate(columns):
        x = 8 + (width + 16) * i
        board.text((x, 4), title, size=TITLE)
        board.paste(colour, (x, 28), (width, height))
        if depth is not None:
            board.paste(depth, (x, 28 + height + 8), (width, height))
    # The empty cell under the photo holds the depth key.
    x, y = 8, 28 + height + 8 + height / 2 - 30
    board.text((x, y), "Depth (scene units)", size=LABEL)
    for i in range(200):
        value = round(255 * (1 - i / 199))
        board.line((x + i, y + 24, x + i, y + 36), fill=(value, value, value))
    board.rectangle((x, y + 24, x + 200, y + 36), outline=GRID)
    board.text((x, y + 42), f"near {depth_range[0]:.2f}", fill=MUTED, size=SMALL)
    board.text((x + 200, y + 42), f"far {depth_range[1]:.2f}", fill=MUTED, size=SMALL, anchor="ra")
    board.text((x, y + 62), "Same range in both depth maps.", fill=MUTED, size=SMALL)
    return board.image


def per_view_change(views, deltas, highlight=None):
    """Columns from a zero line: blue where smoothing helped, red where it hurt."""
    board = Board(760, 250)
    x0, x1, y0, y1 = 48, 750, 44, 214
    limit = max(0.5, float(np.ceil(np.abs(deltas).max() * 2) / 2))

    def level(v):
        return (y0 + y1) / 2 - v / limit * (y1 - y0) / 2

    board.text((0, 4), "Change from smoothing, per test view (dB)", size=TITLE)
    for label, colour, x in [("better with smoothing", BLUE, 430), ("worse with smoothing", RED, 590)]:
        board.bar((x, 10, x + 12, 22), fill=colour, radius=2, end="top")
        board.text((x + 18, 9), label, fill=MUTED, size=SMALL)
    for tick in np.arange(-limit, limit + 1e-9, limit / 2):
        board.line((x0, level(tick), x1, level(tick)), fill=GRID if tick else MUTED, width=1 if tick else 1.2)
        board.text((x0 - 6, level(tick)), f"{tick:+.2f}" if tick else "0", fill=MUTED, size=SMALL, anchor="rm")
    step = (x1 - x0) / len(views)
    bar = min(18, step - 6)
    for i, (view, delta) in enumerate(zip(views, deltas)):
        cx = x0 + step * (i + 0.5)
        board.bar((cx - bar / 2, level(0), cx + bar / 2, level(delta)), fill=BLUE if delta > 0 else RED,
                  end="top" if delta > 0 else "bottom")
        board.text((cx, y1 + 6), str(view), fill=INK if view == highlight else MUTED, size=SMALL, anchor="ma")
    board.text(((x0 + x1) / 2, y1 + 22), "test view", fill=MUTED, size=SMALL, anchor="ma")
    for index in (int(np.argmax(deltas)), int(np.argmin(deltas))):
        cx = x0 + step * (index + 0.5)
        value = deltas[index]
        board.text((cx, level(value) + (-4 if value > 0 else 4)), f"{value:+.2f}", fill=INK, size=SMALL,
                   anchor="md" if value > 0 else "ma")
    return board.image


def small(image_path, width=400):
    """A small copy of a large photo, made in memory."""
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        return source.resize((width, round(source.height * width / source.width)), Image.Resampling.LANCZOS)
