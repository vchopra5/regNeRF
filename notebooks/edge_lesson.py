"""Small visual experiments for Notebook 3. NumPy and Pillow only."""
import numpy as np
from PIL import Image
from notebooks.toy_figures import Board, TITLE, LABEL, SMALL, INK, GRID, ACCENT, TEAL


def pair_figure(strength, connected):
    weight = float(connected)
    gap = 2 / (1 + 2 * strength * weight)
    fitted = (3 - gap/2, 3 + gap/2)
    board = Board(806, 348)
    for col, (label, depths) in enumerate((("Known scene", (2., 4.)), ("Fit with smoothing", fitted))):
        left = 10 + col * 400
        board.text((left, 8), label, size=TITLE)
        board.text((left, 36), "Adjacent image pixels", size=SMALL)
        for x, color in ((left+145, ACCENT), (left+215, TEAL)):
            board.rectangle((x-17, 58, x+17, 86), fill=color)
        if col == 1:
            board.line((left+162, 72, left+198, 72), fill=ACCENT if connected else GRID, width=3)
            board.text((left+180, 99), "linked" if connected else "not linked", anchor="mt", size=SMALL)
        for depth in (2, 3, 4):
            y = 148 + (depth-2)*62
            board.line((left+65, y, left+340, y), fill=GRID)
            board.text((left+48, y), str(depth), anchor="rm", size=SMALL)
        board.text((left, 120), "Depth (scene units)", size=SMALL)
        for x, depth, truth, color in ((left+145, depths[0], 2, ACCENT), (left+215, depths[1], 4, TEAL)):
            y = 148+(depth-2)*62
            board.line((x, 86, x, y), fill=color)
            board.line((x-25, 148+(truth-2)*62, x+25, 148+(truth-2)*62), fill=INK, width=2)
            board.dot(x, y, 7, fill=color)
            board.text((x+16, y-4), f"{depth:.2f}", size=SMALL)
        board.text((left, 300), f"Depth gap: {depths[1]-depths[0]:.2f} scene units", size=TITLE)
    board.text((10, 330), "Black marks = true surfaces. Coloured dots = depth. Same scale in both panels.", size=SMALL)
    return board.image, gap


def ramp_figure(span):
    step = np.r_[np.zeros(4), np.ones(4)]
    ramp = np.clip((np.arange(8)-(7-span)/2)/span, 0, 1)
    board = Board(806, 270)
    for i, (title, values) in enumerate((("One sharp step", step), (f"Change spread over {span} pairs", ramp))):
        left = 10 + i*400
        board.text((left, 8), title, size=TITLE)
        for depth in (0, .5, 1):
            y = 168-100*depth
            board.line((left+35, y, left+365, y), fill=GRID)
            board.text((left+25, y), f"{depth:g}", anchor="rm", size=SMALL)
        points = [(left+40+j*44, 168-v*100) for j, v in enumerate(values)]
        board.line(points, fill=ACCENT, width=3)
        for x, y in points:
            board.dot(x, y, 3, fill=ACCENT)
        differences = np.diff(values)
        board.text((left+35, 191), "Image pixels >   (depth change = 1)", size=SMALL)
        board.text((left+35, 219), f"L2 cost: {np.square(differences).sum():.3f}", size=LABEL)
        board.text((left+205, 219), f"L1 cost: {np.abs(differences).sum():.3f}", size=LABEL)
    board.text((10, 251), "Fixed endpoints. These are penalty costs only, not a fitted NeRF.", size=SMALL)
    return board.image


def tridiagonal(lower, diagonal, upper, rhs):
    """Solve a small positive definite tridiagonal system."""
    d = np.asarray(diagonal, dtype=float).copy()
    b = np.asarray(rhs, dtype=float).copy()
    for i in range(1, len(d)):
        factor = lower[i-1]/d[i-1]
        d[i] -= factor*upper[i-1]
        b[i] -= factor*b[i-1]
    x = b/d
    for i in range(len(d)-2, -1, -1):
        x[i] = (b[i]-upper[i]*x[i+1])/d[i]
    return x


def fit_depth(observed, strength, weights, penalty):
    if strength == 0:
        return observed.copy()
    if penalty == "L2":
        w = strength*weights
        return tridiagonal(-w, 1+np.r_[w, 0]+np.r_[0, w], -w, observed)
    # ADMM solves sum((d-y)^2) + strength * sum(w * abs(diff(d))).
    rho = 4.
    d = observed.copy()
    z, u = np.diff(d), np.zeros(len(d)-1)
    diagonal = 2+rho*np.r_[1., np.full(len(d)-2, 2.), 1.]
    off = np.full(len(d)-1, -rho)
    for _ in range(500):
        v = z-u
        transpose = np.r_[-v[0], v[:-1]-v[1:], v[-1]]
        d = tridiagonal(off, diagonal, off, 2*observed+rho*transpose)
        delta = np.diff(d)
        prior = z.copy()
        z = np.sign(delta+u)*np.maximum(np.abs(delta+u)-strength*weights/rho, 0)
        u += delta-z
        if np.max(np.abs(delta-z)) < 1e-7 and np.max(np.abs(z-prior)) < 1e-7:
            break
    return d


def profile_experiment(case, penalty, strength):
    x = np.arange(64)
    truth = np.where(x < 32, 2., 3.)
    guide = np.where(x < 32, .2, .8)
    if case == "Texture edge":
        guide[10:20] = .8
    elif case == "Missing colour edge":
        guide[:] = .5
    observed = truth + np.random.default_rng(42).normal(0, .12, 64)
    weights = np.exp(-10*np.abs(np.diff(guide)))
    fits = [fit_depth(observed, strength, w, penalty) for w in (np.ones(63), weights)]
    boundary = np.abs(x-31.5) <= 3.5
    flat = np.abs(x-31.5) >= 12.5
    rows = []
    board = Board(806, 420)
    for i, (label, fit) in enumerate(zip(("Uniform weights", "Colour weights"), fits)):
        left = 10+i*400
        board.text((left, 8), f"{penalty}: {label}", size=TITLE)
        px = lambda j: left+40+j*5.1
        py = lambda d: 233-(d-1.6)*112
        board.rectangle((px(28), 52, px(35), 234), fill="#fff1e5")
        for depth in (2., 2.5, 3.):
            board.line((left+35, py(depth), left+365, py(depth)), fill=GRID)
            board.text((left+28, py(depth)), f"{depth:g}", anchor="rm", size=SMALL)
        for j, value in enumerate(observed):
            board.dot(px(j), py(value), 1.5, fill="#b4bbc1")
        for vector, color in ((truth, INK), (fit, TEAL)):
            board.line([(px(j), py(v)) for j, v in enumerate(vector)], fill=color, width=2)
        board.text((left+40, 244), "Pixels 0 to 63 >", size=SMALL)
        b = float(np.sqrt(np.mean((fit[boundary]-truth[boundary])**2)))
        f = float(np.sqrt(np.mean((fit[flat]-truth[flat])**2)))
        rows.append({"Weights": label, "Boundary RMSE": b, "Flat RMSE": f})
        board.text((left+35, 273), f"Boundary RMSE {b:.3f}  |  Flat RMSE {f:.3f}", size=LABEL)
    rgb = np.repeat(np.rint(guide[None, :, None]*255).astype(np.uint8), 3, axis=2)
    board.text((10, 310), "Colour guide", size=SMALL)
    board.paste(Image.fromarray(rgb), (145, 308), (640, 18))
    pixels = np.rint(weights[None, :]*255).astype(np.uint8)
    board.text((10, 342), "Pair weights", size=SMALL)
    board.paste(Image.fromarray(pixels), (145, 340), (640, 18))
    board.text((10, 375), "Black line = reference; teal = fit; grey dots = observations. Orange band = boundary score region.", size=SMALL)
    board.text((10, 396), "Dark pair weights reduce smoothing. Same observations, strength, and depth scale in both panels.", size=SMALL)
    return board.image, rows


def pipeline_figure():
    board = Board(806, 192)
    items = [("Rendered colour", "small image patch"), ("Blur + Sobel", "find colour changes"),
             ("Smoothing weights", "smaller at edges"), ("L1 depth penalty", "uses these weights")]
    for i, (title, subtitle) in enumerate(items):
        left = 8+i*202
        board.rectangle((left, 36, left+184, 113), fill="#f3f5f6")
        board.text((left+9, 51), title, size=LABEL)
        board.text((left+9, 80), subtitle, size=SMALL)
        if i < 3:
            board.text((left+187, 68), ">", size=LABEL)
    board.text((8, 135), "The colour guide sets the weights. This depth penalty sends no gradients back through the guide.", size=SMALL)
    board.text((8, 162), "RegNeRF + edge L1 = RGB loss + original L2 loss + 0.1 x this fine-level L1 loss.", size=SMALL)
    return board.image
