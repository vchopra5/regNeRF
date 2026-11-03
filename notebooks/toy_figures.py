"""Draw the same depth scale for the saved figures and the live pixel inspector."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size=15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def reproject_colours(rgb, depth, foreground, focal, shift):
    """Move visible RGB samples to a parallel camera. Nearest depth wins collisions."""
    y, x = np.nonzero(foreground)
    z = depth[y, x]
    target_x = np.rint(x - focal * shift / z).astype(int)
    inside = (target_x >= 0) & (target_x < rgb.shape[1])
    y, x, z, target_x = y[inside], x[inside], z[inside], target_x[inside]
    index = y * rgb.shape[1] + target_x
    closest = np.full(rgb.shape[:2], np.inf)
    np.minimum.at(closest.ravel(), index, z)
    keep = z == closest[y, target_x]
    image = np.full_like(rgb, [232, 236, 239])
    image[y[keep], target_x[keep]] = rgb[y[keep], x[keep]]
    return image


STRIPE_POINT = (83, 67)


def shear_patch(rgb, truth, prediction, focal, shift):
    """Project one textured patch twice. Both panels track the same reference point."""
    col, row = STRIPE_POINT
    side = 24
    x0, y0 = col - side//2, row - side//2
    x, y = np.meshgrid(np.arange(side+1), np.arange(side+1))
    crop = np.s_[y0:y0+side+1, x0:x0+side+1]
    source = Image.fromarray(rgb).resize((200, 150))
    ImageDraw.Draw(source).rectangle((x0, y0, x0+side, y0+side), outline="white", width=2)
    ImageDraw.Draw(source).ellipse((col-2,row-2,col+2,row+2),fill="white",outline="#203039")
    board = Image.new("RGB", (780, 370), "white")
    board.paste(source, (0, 25))
    draw = ImageDraw.Draw(board)
    draw.text((10, 6), "Original photo", fill="#263c49")
    # Plan view in camera coordinates. Keep the same scale along both axes.
    z = truth[row, col]
    patch_x = (col + .25 - rgb.shape[1]/2) * z / focal
    origin = (100, 290)
    camera = (100 + 31*shift, 290)
    target = (100 + 31*patch_x, 290 - 31*z)
    draw.text((10, 176), "Camera path from above", fill="#263c49")
    draw.line((55,290,145,290),fill="#b9c4c9")
    draw.line([origin,target],fill="#a1acb2",width=2)
    draw.line([camera,target],fill="#00766f",width=2)
    draw.ellipse((target[0]-3,target[1]-3,target[0]+3,target[1]+3),fill="#a94126")
    draw.ellipse((97,287,103,293),outline="#58686f")
    draw.rectangle((camera[0]-4,286,camera[0]+4,294),fill="#00766f")
    draw.text((12, 324), "Circle: start camera", fill="#58686f", font=font(13))
    draw.text((12, 344), f"Square: camera at {shift:+.2f}", fill="#00766f", font=font(13))
    delta = np.degrees(np.arctan2(patch_x-shift,z)-np.arctan2(patch_x,z))
    draw.text((12,302), f"Line of sight turns {abs(delta):.1f} degrees",fill="#263c49")
    common_motion = focal * shift / truth[y0+side//2, x0+side//2]
    for idx, (label, depth) in enumerate([("Known depth", truth), ("Model depth", prediction)]):
        z = depth[crop]
        panel = Image.new("RGB", (285, 210), "#f3f5f6")
        pen = ImageDraw.Draw(panel)
        px = 28 + 8*(x - focal*shift/z + common_motion)
        py = 9 + 8*y
        def vertex(row, col):
            return float(px[row,col]), float(py[row,col])
        # Draw the far cells first wherever a fold hides part of the patch.
        order = np.argsort(-z[:-1,:-1], axis=None) if shift else np.arange(side*side)
        for row, col in zip(*np.unravel_index(order, (side,side))):
            points = [vertex(row,col),vertex(row,col+1),vertex(row+1,col+1),vertex(row+1,col)]
            pen.polygon(points, fill=tuple(rgb[y0+row,x0+col]))
        for j in range(0, side+1, 4):
            pen.line([vertex(k,j) for k in range(side+1)], fill="#a7dae2", width=1)
            pen.line([vertex(j,k) for k in range(side+1)], fill="#a7dae2", width=1)
        dot_x,dot_y = vertex(side//2,side//2)
        guide_x = 28 + 8*(side//2)
        pen.line((guide_x, 0, guide_x, 209), fill="white", width=1)
        pen.ellipse((dot_x-4,dot_y-4,dot_x+4,dot_y+4),fill="white",outline="#203039")
        offset=204+idx*288
        board.paste(panel,(offset,58))
        draw.text((offset+6,36),label,fill="#263c49")
    marker_x, marker_y = STRIPE_POINT
    offset = focal*shift*(1/truth[marker_y,marker_x] - 1/prediction[marker_y,marker_x])
    draw.text((215, 290), f"Marked point error: {offset:+.1f} source pixels", fill="#263c49", font=font(18))
    draw.text((215, 321), "White line: correct point position. Both close-ups follow the true point.", fill="#263c49", font=font(13))
    draw.text((215, 346), "Same paint colours; only depth changes. Close-ups use 8x magnification.", fill="#263c49", font=font(13))
    return board


def depth_image(depth, ids):
    # Mask to the true foreground so that surface depths can be compared.
    shade = np.rint(255 * np.clip((5 - np.where(ids, depth, 5)) / 2.5, 0, 1)).astype("uint8")
    rgb = np.stack([shade, shade, shade], axis=-1)
    rgb[ids == 0] = [36, 49, 60]
    return Image.fromarray(rgb)


def curve_fit_figure(reference, readings, fitted):
    """Compare two fits on identical axes. Grey dots are the fitting data."""
    board = Image.new("RGB", (800, 264), "white")
    draw = ImageDraw.Draw(board)
    low = np.floor(readings.min()*10)/10
    high = np.ceil(readings.max()*10)/10
    for i, (title, curve) in enumerate([("Without smoothing", readings), ("Your fit", fitted)]):
        left = 40 + 398*i
        draw.text((left, 8), title, fill="#203743")
        def points(values):
            return [(left+342*j/(len(values)-1), 207-165*(v-low)/(high-low)) for j,v in enumerate(values)]
        for tick in np.arange(low, high+.01, .2):
            y = points([tick,tick])[0][1]
            draw.line((left,y,left+342,y), fill="#dce2e4")
            draw.text((left-6,y), f"{tick:.1f}", anchor="rm", fill="#455762")
        for x,y in points(readings):
            draw.ellipse((x-2,y-2,x+2,y+2),fill="#a8afb3")
        draw.line(points(reference),fill="#253d50",width=4)
        draw.line(points(curve),fill="#c45324",width=2)
        draw.text((left,215), "Left side",fill="#455762")
        draw.text((left+342,215), "Right side",anchor="ra",fill="#455762")
    for x, label, colour in [(40,"Simulated readings","#828b91"),(222,"True sphere","#253d50"),(405,"Fitted depth","#c45324")]:
        draw.text((x,244),label,fill=colour)
    draw.text((600,244),"Depth in scene units",fill="#455762")
    return board


def patch_experiment(reference, distance, whole_patch):
    """Show the exact unit-opacity loss and the four pairs around the marked pixel."""
    reference = reference.astype(np.float64)
    edited = reference.copy()
    if whole_patch:
        edited += distance
    else:
        edited[4, 4] += distance

    def differences(patch):
        anchor = patch[:-1, :-1]
        return np.stack((anchor - patch[:-1, 1:], anchor - patch[1:, :-1]))

    before, after = differences(reference), differences(edited)
    losses = [float(np.square(delta).sum(axis=0).mean()) for delta in (before, after)]
    changed = ~np.isclose(before, after, atol=1e-12, rtol=0)
    change = losses[1] - losses[0]
    if abs(change) < 1e-12:
        change = 0.0
    board = Image.new("RGB", (1000, 600), "white")
    pen = ImageDraw.Draw(board)
    ink, accent, green = "#203743", "#b44822", "#00766f"
    colour = accent if change else green
    pen.text((20, 12), "Smoothness loss: original  ->  changed", fill=ink, font=font(20))
    pen.text((20, 43), f"{losses[0]:.8f}  ->  {losses[1]:.8f}", fill=ink, font=font(23))
    pen.text((590, 12), f"LOSS CHANGE: {change:+.8f}" if change else "LOSS CHANGE: 0", fill=colour, font=font(24))
    pen.text((590, 48), f"{int(changed.sum())} neighbour pairs changed", fill=colour, font=font(18))
    pen.text((20, 84), f"{'Whole patch' if whole_patch else 'Marked pixel'} moved by {distance:.2f} scene units", fill=ink, font=font(16))
    # Use a fixed loss scale for all distances and both controls.
    bar_max = losses[0] + 4 * .30**2 / 49 + .0002
    for y, label, loss, c in [(85, "Original", losses[0], ink), (107, "Changed", losses[1], colour)]:
        pen.text((590, y), label, fill=c, font=font(13))
        pen.rectangle((660, y, 974, y+12), fill="#e8eef0")
        pen.rectangle((660, y, 660 + 314*loss/bar_max, y+12), fill=c)
    pen.text((20, 132), "8 x 8 depth patch", fill=ink, font=font(17))
    cell = 25
    def centre(row, col):
        return 35 + col*cell + cell//2, 166 + row*cell + cell//2
    for row, col in np.ndindex(edited.shape):
        x, y = centre(row, col)
        tint = np.clip((edited[row,col] - reference.min()) / .32, 0, 1)
        c = tuple(round(a + tint*(b-a)) for a,b in zip((238,244,246),(54,107,134)))
        pen.rectangle((x-11,y-11,x+11,y+11), fill=c)
    pairs = [(0,4,3,"Left"), (0,4,4,"Right"), (1,3,4,"Above"), (1,4,4,"Below")]
    for axis, row, col, label in pairs:
        end = (row+int(axis==1), col+int(axis==0))
        pen.line((centre(row,col), centre(*end)), fill=accent if changed[axis,row,col] else green, width=4)
        x,y = centre(*end) if (row,col)==(4,4) else centre(row,col)
        pen.text((x-4,y-7), label[0], fill=ink, font=font(12))
    x,y = centre(4,4)
    pen.ellipse((x-5,y-5,x+5,y+5),fill=ink)
    pen.text((20, 377), "Dot: moved pixel in single-pixel mode", fill=ink, font=font(13))
    pen.text((20, 399), "Orange: changed pair. Green: unchanged pair.", fill=ink, font=font(13))

    pen.text((320, 132), "Depth across the marked row (same scale in both modes)", fill=ink, font=font(17))
    low = float(reference.min()) - .025
    high = low + .38
    def level(z):
        return 355 - (z-low)*185/(high-low)
    for tick in np.linspace(low+.025, high-.025, 4):
        y = level(tick)
        pen.line((370,y,970,y),fill="#dde4e7")
        pen.text((315,y-8),f"{tick:.2f}",fill=ink,font=font(13))
    for patch,c,width in ((reference,ink,5),(edited,accent,2)):
        points = [(375+84*col,level(z)) for col,z in enumerate(patch[4])]
        pen.line(points,fill=c,width=width)
        for x,y in points:
            pen.ellipse((x-3,y-3,x+3,y+3),fill=c)
    pen.text((370, 377), "Dark: original depth. Orange: changed depth.", fill=ink, font=font(14))
    pen.text((370, 399), "Smoothing sees differences, not absolute distance.", fill=colour, font=font(17))
    pen.line((20,432,978,432),fill="#dce2e4")
    pen.text((20,443), "Pair", fill=ink,font=font(15))
    pen.text((160,443), "Depth difference: original -> changed", fill=ink,font=font(15))
    pen.text((615,443), "Loss contribution: original -> changed", fill=ink,font=font(15))
    for i,(axis,row,col,label) in enumerate(pairs):
        old,new = before[axis,row,col],after[axis,row,col]
        y = 470+24*i
        pen.text((20,y),label,fill=ink,font=font(14))
        pen.text((160,y),f"{old:+.5f} -> {new:+.5f}",fill=accent if changed[axis,row,col] else green,font=font(14))
        pen.text((615,y),f"{old**2/49:.8f} -> {new**2/49:.8f}",fill=ink,font=font(14))
    pen.text((20,576), "Each pair contributes its squared difference / 49. The top loss includes all 98 pairs. Opacity A = 1 here.", fill=ink,font=font(13))
    return board, int(changed.sum()), float(np.abs(edited-reference).mean())


def depth_diagnosis(truth, prediction, ids, x, y):
    """Mark the same ray on both depth maps and the signed error map."""
    foreground = ids > 0
    error = np.zeros_like(prediction)
    error[foreground] = prediction[foreground] - truth[foreground]
    # Use a fixed colour range so changing models preserves the meaning of colour.
    limit = 0.5
    fraction = np.minimum(np.abs(error) / limit, 1)[..., None]
    hue = np.where(error[..., None] < 0, [36, 103, 166], [184, 74, 28])
    rgb = (1 - fraction) * 249 + fraction * hue
    rgb[~foreground] = [36, 49, 60]
    maps = [depth_image(truth, ids), depth_image(prediction, ids),
            Image.fromarray(rgb.astype("uint8"))]
    board = Image.new("RGB", (780, 245), "white")
    pen = ImageDraw.Draw(board)
    for i, (title, panel) in enumerate(zip(("Known depth", "Predicted depth", "Prediction minus reference"), maps)):
        origin = 260*i + 5
        pen.text((origin, 5), title, fill="#233746")
        board.paste(panel.resize((250, 188), Image.Resampling.NEAREST), (origin, 27))
        cx, cy = origin + (x+.5)*250/panel.width, 27 + (y+.5)*188/panel.height
        for radius, colour in [(6, "black"), (5, "white")]:
            pen.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=colour)
    pen.text((5, 222), "Light = near (2.5)     Dark = far (5.0)", fill="#233746")
    pen.text((525, 222), "Near -0.5", fill="#2467a6")
    pen.text((620, 222), "0", fill="#444444")
    pen.text((694, 222), "Far +0.5", fill="#b84a1c")
    return board


def depth_profile(truth, predicted, ids, row, left, right, limits=(2, 6), curve_label="Model prediction"):
    """Show the selected pixels on the reference map and plot both depth profiles."""
    board = Image.new("RGB", (785, 265), "white")
    draw = ImageDraw.Draw(board)
    inset = depth_image(truth, ids).resize((240, 180))
    board.paste(inset, (4, 32))
    draw.text((4, 10), "Reference depth / selected pixels", fill="black")
    draw.line((4+1.2*left, 32+1.2*row, 4+1.2*right, 32+1.2*row), fill="#ffb541", width=3)
    x0,x1,y0,y1 = 296,752,33,210
    draw.text((x0, 9), "Depth (scene units)", fill="black")
    def y(z):
        return y1 - (z-limits[0])*(y1-y0)/(limits[1]-limits[0])
    for z in np.linspace(*limits, 5):
        draw.line((x0,y(z),x1,y(z)),fill="#dce2e4")
        draw.text((x0-5,y(z)),f"{z:.2f}",anchor="rm",fill="#41525a")
    for array,colour in [(truth,"#182c36"),(predicted,"#b9581d")]:
        prev = None
        for col in range(left,right+1):
            if not ids[row,col]:
                prev = None
                continue
            point=(x0+(col-left)*(x1-x0)/max(1,right-left),y(array[row,col]))
            if prev is not None:
                draw.line([prev,point],fill=colour,width=2)
            draw.ellipse((point[0]-2,point[1]-2,point[0]+2,point[1]+2),fill=colour)
            prev=point
    draw.text((x0,218),str(left),fill="black")
    draw.text((x1-15,218),str(right),fill="black")
    draw.text((445,218),f"Pixel column on row {row}",fill="black")
    draw.text((x0,243),"Reference",fill="#182c36")
    draw.text((x0+100,243),curve_label,fill="#b9581d")
    return board


def comparison(scene, folders):
    with np.load(scene/"reference.npz") as data:
        truth,ids=data["depth"],data["ids"]
    board=Image.new("RGB",(220*(1+len(folders)),390),"white")
    draw=ImageDraw.Draw(board)
    for i,(name,folder) in enumerate([("Reference",None),*folders]):
        if folder is None:
            rgb=Image.open(scene/"reference.png")
            depth=truth
        else:
            rgb=Image.open(folder/"rgb.png")
            with np.load(folder/"prediction.npz") as data:
                depth=data["depth"]
        x=10+220*i
        draw.text((x,5),name,fill="black",font=font(15))
        board.paste(rgb,(x,25))
        board.paste(depth_image(depth,ids),(x,185))
    draw.text((10,348),"Depth: light = near (2.5); dark = far (5.0 scene units).",fill="#203743",font=font(13))
    draw.text((10,370),"Depth uses the true foreground mask. RGB includes the background.",fill="#203743",font=font(12))
    return board


def surface_inspector(truth, prediction, ids, focal, angle, row, left, right):
    """Turn the red sphere's visible surface using the same coordinates and scale."""
    height, width = truth.shape
    v,u = np.nonzero(ids == 1)
    theta = np.deg2rad(angle)
    # Saved images and rays use every second full-resolution pixel.
    true_x = (u+.25-width/2)*truth[v,u]/focal
    true_y = (v+.25-height/2)*truth[v,u]/focal
    centre_x,centre_y,centre_z = np.mean(true_x),np.mean(true_y),np.mean(truth[v,u])
    board = Image.new("RGB", (800, 330), "white")
    draw = ImageDraw.Draw(board)
    for i,(label,depth) in enumerate([("True red sphere",truth),("Three-photo baseline",prediction)]):
        z = depth[v,u]
        x = (u+.25-width/2)*z/focal
        y = (v+.25-height/2)*z/focal
        horizontal = (x-centre_x)*np.cos(theta)+(z-centre_z)*np.sin(theta)
        distance = -(x-centre_x)*np.sin(theta)+(z-centre_z)*np.cos(theta)
        px = 200+i*400+150*horizontal
        py = 161+150*(y-centre_y)
        shade = np.clip((3.65-z)/.85,0,1)
        ink = np.column_stack([43+151*shade,67+149*shade,79+143*shade]).astype(int)
        selected = (v==row)&(u>=left)&(u<=right)
        ink[selected] = (237,117,24)
        for j in np.argsort(-distance):
            draw.ellipse((px[j]-1.5,py[j]-1.5,px[j]+1.5,py[j]+1.5),fill=tuple(ink[j]))
        draw.text((i*400+100,12),label,fill="#20363d",font=font(18))
    draw.text((20,296),"Orange: the same source pixels on both surfaces. Same scale and rotation.",fill="#334953",font=font(15))
    draw.text((20,316),"Only the surface visible to the held-out camera is shown. This is not a full mesh.",fill="#334953",font=font(13))
    return board
