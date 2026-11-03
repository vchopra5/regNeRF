"""Figures for the toy lesson.

Every figure is drawn at SCALE times its display size, then shown at the display size.
This keeps text and lines sharp. Model renders are enlarged with nearest-neighbour
sampling, so each predicted pixel stays visible instead of being blurred.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCALE = 2

# One type scale and one palette for every figure.
TITLE, LABEL, SMALL = 15, 13, 11
INK, MUTED, GRID = "#203743", "#5b6b74", "#dde4e7"
ACCENT, TEAL, MARK = "#c0501f", "#00766f", "#f0a030"
PANEL, BACKDROP = "#f3f5f6", "#24313c"

STRIPE_POINT = (83, 67)
NEAR, FAR = 2.5, 5.0


def font(size=LABEL):
    """Return Pillow's bundled font, so the figures look the same on every system."""
    return ImageFont.load_default(size=size)


def _scaled(value):
    if isinstance(value, tuple):
        return tuple(_scaled(v) for v in value)
    if isinstance(value, list):
        return [_scaled(v) for v in value]
    return float(value) * SCALE


class Board:
    """A white canvas in display units. It draws at SCALE times the resolution."""

    def __init__(self, width, height, background="white"):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width * SCALE, height * SCALE), background)
        self._pen = ImageDraw.Draw(self.image)

    def text(self, xy, text, fill=INK, size=LABEL, anchor=None):
        self._pen.text(_scaled(xy), text, fill=fill, font=font(size * SCALE), anchor=anchor)

    def line(self, xy, fill=INK, width=1):
        self._pen.line(_scaled(xy), fill=fill, width=max(1, round(width * SCALE)))

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self._pen.rectangle(_scaled(xy), fill=fill, outline=outline, width=max(1, round(width * SCALE)))

    def dot(self, x, y, radius, fill=None, outline=None, width=1):
        self._pen.ellipse(_scaled((x - radius, y - radius, x + radius, y + radius)), fill=fill,
                          outline=outline, width=max(1, round(width * SCALE)))

    def polygon(self, points, fill):
        self._pen.polygon(_scaled(points), fill=fill)

    def bar(self, xy, fill, radius=4, end="right"):
        """A bar rounded only at its value end; the baseline end stays square."""
        x0, y0, x1, y1 = xy
        if abs(x1 - x0) < 0.5 or abs(y1 - y0) < 0.5:
            return
        corners = {"right": (False, True, True, False), "top": (True, True, False, False),
                   "bottom": (False, False, True, True)}[end]
        radius = min(radius, abs(x1 - x0) / 2, abs(y1 - y0) / 2)
        self._pen.rounded_rectangle(_scaled((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))),
                                    radius=radius * SCALE, fill=fill, corners=corners)

    def paste(self, picture, xy, size):
        """Paste an image at a display position and size. Keep its pixels sharp."""
        picture = picture.convert("RGB").resize((size[0] * SCALE, size[1] * SCALE), Image.Resampling.NEAREST)
        self.image.paste(picture, (round(xy[0] * SCALE), round(xy[1] * SCALE)))


def crisp(picture, width):
    """Enlarge a small render to a display width with visible, unblurred pixels."""
    picture = Image.open(picture) if not isinstance(picture, Image.Image) else picture
    picture = picture.convert("RGB")
    height = round(picture.height * width / picture.width)
    return picture.resize((width * SCALE, height * SCALE), Image.Resampling.NEAREST)


def depth_image(depth, ids):
    """Grey depth map of the true foreground: light is near, dark is far."""
    shade = np.rint(255 * np.clip((FAR - np.where(ids, depth, FAR)) / (FAR - NEAR), 0, 1)).astype("uint8")
    rgb = np.stack([shade, shade, shade], axis=-1)
    rgb[ids == 0] = (36, 49, 60)
    return Image.fromarray(rgb)


def _depth_legend(board, x, y, width):
    for i in range(width):
        value = round(255 * (1 - i / (width - 1)))
        board.line((x + i, y, x + i, y + 10), fill=(value, value, value))
    board.rectangle((x, y, x + width, y + 10), outline=GRID)
    board.text((x, y + 14), f"near ({NEAR})", fill=MUTED, size=SMALL)
    board.text((x + width, y + 14), f"far ({FAR})", fill=MUTED, size=SMALL, anchor="ra")


def comparison(scene, folders, image_width=300):
    """Reference and model columns: colour on top, depth below, one depth scale."""
    with np.load(scene / "reference.npz") as data:
        truth, ids = data["depth"], data["ids"]
    columns = [("Reference", None), *folders]
    height = round(image_width * 3 / 4)
    board = Board(16 + (image_width + 16) * len(columns), 26 + 2 * height + 12 + 56)
    for i, (name, folder) in enumerate(columns):
        x = 8 + (image_width + 16) * i
        if folder is None:
            rgb, depth = Image.open(scene / "reference.png"), truth
        else:
            rgb = Image.open(folder / "rgb.png")
            with np.load(folder / "prediction.npz") as data:
                depth = data["depth"]
        board.text((x, 4), name, size=TITLE)
        board.paste(rgb, (x, 26), (image_width, height))
        board.paste(depth_image(depth, ids), (x, 26 + height + 6), (image_width, height))
    top = 26 + 2 * height + 16
    board.text((8, top), "Depth (scene units):", fill=MUTED, size=SMALL)
    _depth_legend(board, 130, top + 2, 180)
    board.text((8, top + 34), "Depth shows the spheres only. Colour also shows the background.", fill=MUTED, size=SMALL)
    return board.image


def shear_patch(rgb, truth, prediction, focal, shift):
    """Move the camera sideways and project one textured patch with two depth maps."""
    col, row = STRIPE_POINT
    side = 24
    x0, y0 = col - side // 2, row - side // 2
    x, y = np.meshgrid(np.arange(side + 1), np.arange(side + 1))
    crop = np.s_[y0:y0 + side + 1, x0:x0 + side + 1]
    board = Board(730, 330)

    # Left: the held-out photo with the patch marked.
    board.text((8, 4), "Held-out photo", size=TITLE)
    board.paste(Image.fromarray(rgb), (8, 26), (200, 150))
    board.rectangle((8 + x0, 26 + y0, 8 + x0 + side, 26 + y0 + side), outline="white", width=1.5)
    board.dot(8 + col + .5, 26 + row + .5, 2.5, fill="white", outline=INK)

    # Left, lower: plan view in camera coordinates, same scale on both axes.
    board.text((8, 188), "Camera path, seen from above", size=LABEL)
    z = truth[row, col]
    patch_x = (col + .25 - rgb.shape[1] / 2) * z / focal
    ox, oy, unit = 108, 300, 31
    target = (ox + unit * patch_x, oy - unit * z)
    camera = ox + unit * shift
    board.line((ox - 60, oy, ox + 60, oy), fill=GRID)
    board.line([(ox, oy), target], fill="#a1acb2", width=1.5)
    board.line([(camera, oy), target], fill=TEAL, width=1.5)
    board.dot(*target, 3, fill=ACCENT)
    board.dot(ox, oy, 4, outline=MUTED, width=1.5)
    board.rectangle((camera - 4, oy - 4, camera + 4, oy + 4), fill=TEAL)
    board.dot(14, 318, 4, outline=MUTED, width=1.5)
    board.text((24, 311), "start camera", fill=MUTED, size=SMALL)
    board.rectangle((106, 314, 114, 322), fill=TEAL)
    board.text((120, 311), f"camera at {shift:+.2f}", fill=TEAL, size=SMALL)

    # Right: the same paint projected with known depth and with model depth.
    common_motion = focal * shift / truth[y0 + side // 2, x0 + side // 2]
    for idx, (label, depth) in enumerate([("With known depth", truth), ("With model depth", prediction)]):
        left, top = 236 + idx * 248, 26
        board.text((left, 4), label, size=TITLE)
        board.rectangle((left, top, left + 236, top + 210), fill=PANEL)
        zz = depth[crop]
        px = left + 22 + 8 * (x - focal * shift / zz + common_motion)
        py = top + 9 + 8 * y

        def vertex(r, c):
            return float(px[r, c]), float(py[r, c])

        # Draw far cells first wherever a fold hides part of the patch.
        order = np.argsort(-zz[:-1, :-1], axis=None) if shift else np.arange(side * side)
        for r, c in zip(*np.unravel_index(order, (side, side))):
            board.polygon([vertex(r, c), vertex(r, c + 1), vertex(r + 1, c + 1), vertex(r + 1, c)],
                          fill=tuple(int(v) for v in rgb[y0 + r, x0 + c]))
        for j in range(0, side + 1, 4):
            board.line([vertex(k, j) for k in range(side + 1)], fill="#a7dae2")
            board.line([vertex(j, k) for k in range(side + 1)], fill="#a7dae2")
        guide = left + 22 + 8 * (side // 2)
        board.line((guide, top, guide, top + 210), fill="white", width=1.5)
        board.dot(*vertex(side // 2, side // 2), 4, fill="white", outline=INK, width=1.5)
    offset = focal * shift * (1 / truth[row, col] - 1 / prediction[row, col])
    board.text((236, 248), f"Marked point error: {offset:+.1f} pixels", size=TITLE)
    board.text((236, 272), "White line: where the marked point belongs.", fill=MUTED, size=LABEL)
    board.text((236, 292), "Both panels use the same paint. Only depth differs. Close-ups are enlarged 8 times.",
               fill=MUTED, size=SMALL)
    return board.image


def surface_inspector(truth, prediction, ids, focal, angle, row, left, right):
    """Turn the red sphere's visible surface with the same coordinates and scale."""
    height, width = truth.shape
    v, u = np.nonzero(ids == 1)
    theta = np.deg2rad(angle)
    # Saved images and rays use every second full-resolution pixel.
    true_x = (u + .25 - width / 2) * truth[v, u] / focal
    true_y = (v + .25 - height / 2) * truth[v, u] / focal
    centre_x, centre_y, centre_z = np.mean(true_x), np.mean(true_y), np.mean(truth[v, u])
    board = Board(640, 290)
    for i, (label, depth) in enumerate([("True red sphere", truth), ("Three-photo model", prediction)]):
        z = depth[v, u]
        x = (u + .25 - width / 2) * z / focal
        y = (v + .25 - height / 2) * z / focal
        horizontal = (x - centre_x) * np.cos(theta) + (z - centre_z) * np.sin(theta)
        distance = -(x - centre_x) * np.sin(theta) + (z - centre_z) * np.cos(theta)
        px = 160 + i * 320 + 140 * horizontal
        py = 150 + 140 * (y - centre_y)
        shade = np.clip((3.65 - z) / .85, 0, 1)
        ink = np.column_stack([43 + 151 * shade, 67 + 149 * shade, 79 + 143 * shade]).astype(int)
        ink[(v == row) & (u >= left) & (u <= right)] = (237, 117, 24)
        for j in np.argsort(-distance):
            board.dot(px[j], py[j], 1.3, fill=tuple(int(c) for c in ink[j]))
        board.text((160 + i * 320, 6), label, size=TITLE, anchor="ma")
    board.text((320, 272), "Orange: the same strip of pixels on both surfaces. Only the side the camera sees is shown.",
               fill=MUTED, size=SMALL, anchor="ma")
    return board.image


def depth_profile(truth, predicted, ids, row, left, right, limits=(2, 6), curve_label="Model"):
    """Mark the strip on the reference depth map and plot both depth curves along it."""
    board = Board(640, 230)
    board.text((0, 4), "Strip on the depth map", size=LABEL)
    board.paste(depth_image(truth, ids), (0, 26), (200, 150))
    board.line((left, 26 + row + .5, right + 1, 26 + row + .5), fill=MARK, width=2.5)
    x0, x1, y0, y1 = 270, 630, 30, 180
    board.text((x0 - 40, 4), "Depth along the strip (scene units)", size=LABEL)

    def level(z):
        return y1 - (z - limits[0]) * (y1 - y0) / (limits[1] - limits[0])

    for z in np.linspace(*limits, 5):
        board.line((x0, level(z), x1, level(z)), fill=GRID)
        board.text((x0 - 6, level(z)), f"{z:.2f}", fill=MUTED, size=SMALL, anchor="rm")
    for array, colour in [(truth, INK), (predicted, ACCENT)]:
        previous = None
        for col in range(left, right + 1):
            if not ids[row, col]:
                previous = None
                continue
            point = (x0 + (col - left) * (x1 - x0) / max(1, right - left), level(array[row, col]))
            if previous is not None:
                board.line([previous, point], fill=colour, width=2)
            board.dot(*point, 2.2, fill=colour)
            previous = point
    board.text((x0, 188), f"column {left}", fill=MUTED, size=SMALL)
    board.text((x1, 188), f"column {right}", fill=MUTED, size=SMALL, anchor="ra")
    board.line((x0, 214, x0 + 18, 214), fill=INK, width=2.5)
    board.text((x0 + 24, 207), "Reference", size=SMALL)
    board.line((x0 + 100, 214, x0 + 118, 214), fill=ACCENT, width=2.5)
    board.text((x0 + 124, 207), curve_label, fill=ACCENT, size=SMALL)
    return board.image


def curve_fit_figure(reference, readings, fitted):
    """Compare an exact fit with a smoothed fit on identical axes."""
    board = Board(640, 250)
    low = np.floor(readings.min() * 10) / 10
    high = np.ceil(readings.max() * 10) / 10
    for i, (title, curve) in enumerate([("Exact fit (no smoothing)", readings), ("Your smoothed fit", fitted)]):
        left = 36 + 320 * i
        right = left + 270
        board.text((left, 4), title, size=TITLE)

        def points(values, left=left, right=right):
            return [(left + (right - left) * j / (len(values) - 1), 196 - 160 * (v - low) / (high - low))
                    for j, v in enumerate(values)]

        for tick in np.arange(low, high + .01, .2):
            y = points([tick, tick])[0][1]
            board.line((left, y, right, y), fill=GRID)
            board.text((left - 6, y), f"{tick:.1f}", fill=MUTED, size=SMALL, anchor="rm")
        for x, y in points(readings):
            board.dot(x, y, 1.8, fill="#a8afb3")
        board.line(points(reference), fill=INK, width=3.5)
        board.line(points(curve), fill=ACCENT, width=1.8)
        board.text((left, 202), "left edge", fill=MUTED, size=SMALL)
        board.text((right, 202), "right edge", fill=MUTED, size=SMALL, anchor="ra")
    for x, label, colour, kind in [(36, "noisy readings", "#a8afb3", "dot"), (170, "true depth", INK, "line"),
                                   (290, "fit", ACCENT, "line")]:
        if kind == "dot":
            board.dot(x + 6, 232, 3, fill=colour)
        else:
            board.line((x, 232, x + 16, 232), fill=colour, width=3)
        board.text((x + 22, 225), label, fill=MUTED, size=SMALL)
    return board.image


PATCH_PAIRS = [(0, 4, 3, "Left"), (0, 4, 4, "Right"), (1, 3, 4, "Above"), (1, 4, 4, "Below")]


def patch_numbers(reference, distance, whole_patch):
    """Return both patches, both losses, the change mask, and the four pairs around the marked pixel."""
    reference = reference.astype(np.float64)
    edited = reference + distance if whole_patch else reference.copy()
    if not whole_patch:
        edited[4, 4] += distance

    def differences(patch):
        anchor = patch[:-1, :-1]
        return np.stack((anchor - patch[:-1, 1:], anchor - patch[1:, :-1]))

    before, after = differences(reference), differences(edited)
    losses = [float(np.square(d).sum(axis=0).mean()) for d in (before, after)]
    changed = ~np.isclose(before, after, atol=1e-12, rtol=0)
    rows = [{"Pair": label, "Depth difference before": float(before[a, r, c]),
             "Depth difference after": float(after[a, r, c]),
             "Loss share before": float(before[a, r, c] ** 2 / 49), "Loss share after": float(after[a, r, c] ** 2 / 49)}
            for a, r, c, label in PATCH_PAIRS]
    return reference, edited, losses, changed, rows


def patch_experiment(reference, distance, whole_patch):
    """Draw the 8 x 8 patch with the four marked pairs, and the depth across the marked row."""
    reference, edited, losses, changed, _ = patch_numbers(reference, distance, whole_patch)
    board = Board(640, 250)
    board.text((0, 4), "8 x 8 depth patch", size=TITLE)
    cell = 24

    def centre(r, c):
        return 12 + c * cell + cell / 2, 30 + r * cell + cell / 2

    for r, c in np.ndindex(edited.shape):
        x, y = centre(r, c)
        tint = np.clip((edited[r, c] - reference.min()) / .32, 0, 1)
        colour = tuple(round(a + tint * (b - a)) for a, b in zip((238, 244, 246), (54, 107, 134)))
        board.rectangle((x - 11, y - 11, x + 11, y + 11), fill=colour)
    for axis, r, c, label in PATCH_PAIRS:
        end = (r + int(axis == 1), c + int(axis == 0))
        board.line((centre(r, c), centre(*end)), fill=ACCENT if changed[axis, r, c] else TEAL, width=3)
        x, y = centre(*end) if (r, c) == (4, 4) else centre(r, c)
        board.text((x, y), label[0], fill=INK, size=SMALL, anchor="mm")
    if not whole_patch:
        board.dot(*centre(4, 4), 4.5, fill=INK)
    board.line((12, 236, 28, 236), fill=ACCENT, width=3)
    board.text((32, 229), "changed pair", fill=MUTED, size=SMALL)
    board.line((112, 236, 128, 236), fill=TEAL, width=3)
    board.text((132, 229), "unchanged pair", fill=MUTED, size=SMALL)

    x0, x1, y0, y1 = 260, 630, 34, 196
    board.text((x0 - 36, 4), "Depth across the marked row (scene units)", size=TITLE)
    low = float(reference.min()) - .025
    high = low + .38

    def level(z):
        return y1 - (z - low) * (y1 - y0) / (high - low)

    for tick in np.linspace(low + .025, high - .025, 4):
        board.line((x0, level(tick), x1, level(tick)), fill=GRID)
        board.text((x0 - 6, level(tick)), f"{tick:.2f}", fill=MUTED, size=SMALL, anchor="rm")
    for patch, colour, width in ((reference, INK, 4), (edited, ACCENT, 2)):
        points = [(x0 + 10 + (x1 - x0 - 20) * c / 7, level(z)) for c, z in enumerate(patch[4])]
        board.line(points, fill=colour, width=width)
        for point in points:
            board.dot(*point, 3, fill=colour)
    board.line((x0, 236, x0 + 16, 236), fill=INK, width=3)
    board.text((x0 + 22, 229), "original", fill=MUTED, size=SMALL)
    board.line((x0 + 100, 236, x0 + 116, 236), fill=ACCENT, width=3)
    board.text((x0 + 122, 229), "changed", fill=MUTED, size=SMALL)
    return board.image, int(changed.sum()), float(np.abs(edited - reference).mean())
