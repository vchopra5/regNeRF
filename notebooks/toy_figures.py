"""Draw the same depth scale for the saved figures and the live pixel inspector."""
import numpy as np
from PIL import Image, ImageDraw


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
    board = Image.new("RGB", (780, 322), "white")
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
        pen.ellipse((dot_x-4,dot_y-4,dot_x+4,dot_y+4),fill="white",outline="#203039")
        offset=204+idx*288
        board.paste(panel,(offset,58))
        draw.text((offset+6,36),label,fill="#263c49")
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
    for x, label, colour in [(40,"Measured depth","#828b91"),(222,"True sphere","#253d50"),(405,"Fitted depth","#c45324")]:
        draw.text((x,244),label,fill=colour)
    draw.text((600,244),"Depth in scene units",fill="#455762")
    return board


def patch_experiment(reference, distance, whole_patch):
    """Edit a central sphere patch and measure the trainer's unit-opacity loss."""
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
    losses = [np.square(delta).sum(axis=0).mean() for delta in (before, after)]
    changed = ~np.isclose(before, after, atol=1e-12, rtol=0)
    board = Image.new("RGB", (760, 277), "white")
    pen = ImageDraw.Draw(board)
    ink, accent = "#203743", "#b44822"
    pen.text((16, 8), "8 x 8 surface patch", fill=ink)
    cell = 23
    def centre(row, col):
        return 22 + col*cell + cell//2, 36 + row*cell + cell//2
    for row, col in np.ndindex(edited.shape):
        x, y = centre(row, col)
        tint = min(1., max(0., (edited[row,col] - reference.min()) / .3))
        colour = tuple(round(a + tint*(b-a)) for a,b in zip((238,244,246),(54,107,134)))
        pen.rectangle((x-10,y-10,x+10,y+10), fill=colour)
    for axis, row, col in np.argwhere(changed):
        pen.line((centre(row,col), centre(row+int(axis==1),col+int(axis==0))),fill=accent,width=3)
    x,y = centre(4,4)
    pen.ellipse((x-4,y-4,x+4,y+4),fill=ink)
    pen.text((16,228), "Dot: marked pixel", fill=ink)
    pen.text((16,248), "Orange: changed neighbour pair", fill=accent)

    pen.text((285,8), "Depth across the marked row (scene units)", fill=ink)
    bottom = float(np.floor(reference.min()*100)/100) - .02
    def level(depth):
        return 213 - (depth-bottom)*480
    for tick in range(4):
        depth = round(reference.min(),2) + tick*.1
        y = level(depth)
        pen.line((320,y,725,y),fill="#dde4e7")
        pen.text((282,y-6),f"{depth:.2f}",fill=ink)
    for patch,colour in ((reference,ink),(edited,accent)):
        points = [(322+57*col, level(z)) for col,z in enumerate(patch[4])]
        pen.line(points, fill=colour,width=2)
        for x,y in points:
            pen.ellipse((x-3,y-3,x+3,y+3),fill=colour)
    pen.text((315,227),"True depth", fill=ink)
    pen.text((422,227),"Edited depth", fill=accent)
    pen.text((315,248),f"Patch loss: {losses[0]:.7f}  ->  {losses[1]:.7f}",fill=ink)
    return board, int(changed.sum()), np.abs(edited-reference).mean()


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
    board=Image.new("RGB",(210*(1+len(folders)),355),"white")
    draw=ImageDraw.Draw(board)
    for i,(name,folder) in enumerate([("Reference",None),*folders]):
        if folder is None:
            rgb=Image.open(scene/"reference.png")
            depth=truth
        else:
            rgb=Image.open(folder/"rgb.png")
            with np.load(folder/"prediction.npz") as data:
                depth=data["depth"]
        x=5+210*i
        draw.text((x,5),name,fill="black")
        board.paste(rgb,(x,25))
        board.paste(depth_image(depth,ids),(x,185))
    return board


def surface_inspector(truth, prediction, ids, focal, angle, row, left, right):
    """Back-project both depth maps into camera space and turn their visible surfaces."""
    height, width = truth.shape
    v, u = np.nonzero(ids)
    theta = np.deg2rad(angle)
    board = Image.new("RGB", (740, 310), "#f6f7f7")
    draw = ImageDraw.Draw(board)
    for i, (label, depth) in enumerate([("Known surface", truth), ("Predicted surface", prediction)]):
        z = depth[v, u]
        x = (u + 0.5 - width/2) * z / focal
        y = (v + 0.5 - height/2) * z / focal
        # Rotate around the sphere centre, four units from this camera.
        horizontal = x*np.cos(theta) + (z-4)*np.sin(theta)
        distance = -x*np.sin(theta) + (z-4)*np.cos(theta)
        px = 127 + i*250 + 99*horizontal
        py = 141 + 99*y
        shade = np.clip((4.12-z)/1.25, 0, 1)
        ink = np.column_stack([43+151*shade, 67+149*shade, 79+143*shade]).astype(int)
        ink[(v==row)&(u>=left)&(u<=right)] = (237,117,24)
        for j in np.argsort(-distance):
            draw.ellipse((px[j]-1.25,py[j]-1.25,px[j]+1.25,py[j]+1.25),fill=tuple(ink[j]))
        draw.text((i*250+65,12),label,fill="#20363d")
    # The lens shows actual source pixels, at identical magnification for both maps.
    size = max(1,right-left+1)
    top, bottom = max(0,row-max(3,size//5)), min(height,row+max(3,size//5)+1)
    for i,depth in enumerate([truth,prediction]):
        crop=depth_image(depth,ids).crop((left,top,right+1,bottom))
        crop.thumbnail((200,78))
        factor=min(200/crop.width,78/crop.height)
        crop=crop.resize((max(1,round(crop.width*factor)),max(1,round(crop.height*factor))),Image.Resampling.NEAREST)
        board.paste(crop,(510,55+118*i))
        draw.text((510,34+118*i),["Reference pixels","Predicted pixels"][i],fill="#20363d")
    draw.text((18,278),"Orange: selected pixels     |     Only the surface visible to the original camera is sampled.",fill="#334953")
    return board
