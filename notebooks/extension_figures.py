"""Small figures for colour edges, depth boundaries, and narrow openings."""
import numpy as np
from PIL import Image

from canny_mask import canny
from notebooks.lab import extension_case, solve_depth
from notebooks.toy_figures import Board, LABEL, SMALL, TITLE


def canny_examples():
    """Use the same colour and depth signals as the one-dimensional lesson."""
    board = Board(840, 370)
    for x, title in ((8, "Known scene"), (205, "Colour image"), (415, "Canny edges"), (625, "True depth")):
        board.text((x, 7), title, size=TITLE)
    names = ("Aligned edge", "Texture edge", "Missing colour edge")
    for i, name in enumerate(names):
        truth, _, guide, _, _ = extension_case(name)
        rgb = np.repeat(np.rint(guide[None, :, None]*255).astype(np.uint8), 48, axis=0)
        rgb = np.repeat(rgb, 3, axis=2)
        edges = canny(rgb)
        depth = np.repeat(np.rint(64+(truth[None, :]-2)*160).astype(np.uint8), 48, axis=0)
        y = 40 + i*95
        board.text((8, y+12), name, size=LABEL)
        for x, picture in ((205, rgb), (415, edges.astype(np.uint8)*255), (625, depth)):
            board.paste(Image.fromarray(picture).convert("RGB"), (x, y), (195, 73))
    board.text((8, 333), "White in the edge map marks a detected colour edge. Depth: dark = 2; light = 3.", size=SMALL)
    board.text((8, 351), "A colour edge can match a depth boundary, add a false boundary, or miss a real one.", size=SMALL)
    return board.image


def canny_photo(image):
    """Show the actual detector output, with no change to its thresholds."""
    rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.LANCZOS))
    edges = canny(rgb)
    overlay = rgb.copy()
    overlay[edges] = (255, 90, 20)
    board = Board(840, 250)
    for i, (title, array) in enumerate((("Training photo: view 25", rgb), ("Canny edge map", edges.astype(np.uint8)*255), ("Edges over the photo", overlay))):
        board.text((8+i*280, 6), title, size=TITLE)
        board.paste(Image.fromarray(array).convert("RGB"), (8+i*280, 35), (268, 201))
    return board.image


def opening_example(width):
    """Fit a horizontal depth signal with a far wall behind a narrow slot."""
    x = np.arange(128)
    slot = (x >= 64-width//2) & (x < 64+width//2)
    truth = np.where(slot, 4., 2.)
    guide = np.where(slot, .2, .8)
    uniform = solve_depth(truth, 8)
    weighted = solve_depth(truth, 8, np.exp(-10*np.abs(np.diff(guide))))
    return truth, uniform, weighted, slot


def opening_figure(width):
    truth, uniform, weighted, slot = opening_example(width)
    board = Board(840, 525)
    rgb = np.full((80, 128, 3), [222, 166, 62], dtype=np.uint8)
    rgb[12:68, slot] = [35, 71, 78]
    columns = [("Colour illustration", Image.fromarray(rgb))]
    for label, depth in (("Known depth", truth), ("Uniform smoothing", uniform), ("Edge weighted", weighted)):
        field = np.full((80, 128), 2.)
        field[12:68] = depth
        shade = np.rint(224-(field-2)*80).clip(0, 255).astype(np.uint8)
        columns.append((label, Image.fromarray(shade)))
    for i, (label, image) in enumerate(columns):
        left = 8+i*210
        board.text((left, 7), label, size=LABEL)
        board.paste(image, (left, 34), (194, 122))
        # Mark the scanline at the image margins without hiding the signal.
        board.line((left, 95, left+7, 95), fill="#c0501f", width=2)
        board.line((left+187, 95, left+194, 95), fill="#c0501f", width=2)
    board.text((8, 170), "Depth: light = near plate (2); dark = far wall (4). Same scale in all three depth maps.", size=SMALL)
    board.text((8, 198), "Depth across the opening", size=TITLE)
    px = lambda x: 60+(x-40)*748/48
    py = lambda d: 417-(d-2)*170/2
    board.rectangle((px(64-width/2-.5), 237, px(64+width/2-.5), 423), fill="#edf1f3")
    for depth in (2, 3, 4):
        board.line((60, py(depth), 808, py(depth)), fill="#dde4e7")
        board.text((38, py(depth)), str(depth), anchor="rm", size=SMALL)
    for x in (40, 48, 56, 64, 72, 80, 88):
        board.text((px(x), 430), str(x), anchor="mt", size=SMALL)
    for label, curve, colour, offset in (("Known", truth, "#203743", 0),
                                        ("Uniform", uniform, "#c0501f", 145),
                                        ("Edge weighted", weighted, "#00766f", 300)):
        board.text((60+offset, 221), label, fill=colour, size=SMALL)
        board.line([(px(x), py(curve[x])) for x in range(40, 89)], fill=colour,
                   width=4 if label == "Known" else 2)
    board.text((60, 452), "Image column. Shaded region = true opening. Larger depth means farther away.", size=SMALL)
    centre = slice(63, 65)
    board.text((8, 480), f"At the centre: known 4.00 | uniform {uniform[centre].mean():.2f} | edge weighted {weighted[centre].mean():.2f} scene units.", size=LABEL)
    board.text((8, 506), "Exact depth input; strength 8; horizontal fitting only. The RGB guide is held fixed, not trained.", size=SMALL)
    return board.image
