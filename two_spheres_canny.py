"""Compare depth smoothing with and without masks from training photos.

Use compare_canny.py for the comparison on unseen camera patches.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from canny_mask import canny, dilate

import train as training
from data import get_rays
from generate_sphere import orbit_rotation, scan40_rotations
from model import MipNeRF


H, W = 300, 400
FOCAL = 0.5 * H / np.tan(np.deg2rad(18))
SPHERES = [((0.55, -0.35, 0), 0.65), ((-0.7, 0.42, 0), 0.78)]


def scene(rotation, distance=4):
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(rotation)
    pose[:3, 3] = distance * pose[:3, 2]
    rays = get_rays(H, W, FOCAL, pose)
    o, d = (r.numpy() for r in rays[:2])
    a = (d*d).sum(-1)
    depth = np.full((H, W), np.inf)
    ids = np.zeros((H, W), dtype=np.uint8)
    rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    for i, (center, radius) in enumerate(SPHERES, 1):
        v = o - center
        b = (v*d).sum(-1)
        disc = b*b - a*((v*v).sum(-1)-radius**2)
        t = (-b - np.sqrt(disc.clip(0))) / a
        hit = (disc >= 0) & (t > 0) & (t < depth)
        depth[hit], ids[hit] = t[hit], i
        n = (v[hit] + t[hit, None]*d[hit])/radius
        shade = .65 + .35*(n @ np.array([1.,-1.,2.])/np.sqrt(6)).clip(0)
        color = np.where((abs(n[:, 2:3]) < .30), [244,192,71], [199,60,46]) if i == 1 else np.array([50,142,183])
        rgb[hit] = np.rint(shade[:,None]*color)
    return pose, rays, rgb, depth, ids


def main():
    out = Path('checkpoint') / ('two_spheres_canny_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    print(out, flush=True)
    rig = scan40_rotations(Path('data/dtu/scan40'))
    frames, masks = [], []
    for i,(view,rot) in enumerate(rig['train']):
        pose, _, rgb, _, _ = scene(rot,(3.6,4.,4.4)[i])
        Image.fromarray(rgb).save(out/f'train{i}.png')
        masks.append(canny(rgb))
        frames.append(dict(file_path=f'train{i}',transform_matrix=pose.tolist(),dtu_view_id=view))
    (out/'transforms_train.json').write_text(json.dumps(dict(camera_angle_x=2*np.arctan(W/(2*FOCAL)),frames=frames)))
    mask = torch.from_numpy(np.stack(masks))
    for name, guide in [('uniform',torch.zeros_like(mask)),('canny',mask)]:
        training.train(3,out,out/name,.1,stop_after=300,depth_weight_start=.1,
                       regnerf_density=True,training_edge_masks=guide,scale=1)
    report(out, rig, masks)


def panels(items, path):
    sheet = Image.new('RGB',(W*len(items),H+30),'white')
    draw = ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate(items):
        draw.text((W*i+8,9),label,fill='black')
        sheet.paste(Image.fromarray(img).convert('RGB'),(W*i,30))
    sheet.save(path)


@torch.inference_mode()
def report(out,rig,masks):
    dest = Path('renders') / out.name
    dest.mkdir(parents=True,exist_ok=True)
    rgb = np.asarray(Image.open(out/'train0.png')).copy()
    overlay = rgb.copy()
    overlay[masks[0]] = [255,0,225]
    panels([('Training view 25',rgb),('Canny mask used in training',255*masks[0].astype('uint8')),
            ('Mask overlay',overlay)],dest/'mask.png')
    models = []
    for name in ('uniform','canny'):
        model = MipNeRF().cuda().eval()
        model.load_state_dict(torch.load(out/name/'mipnerf_final.pt',map_location='cpu'))
        models.append(model)
    records = []
    for view, rot in [('demo',orbit_rotation(15,5)),*rig['test']]:
        _,rays,rgb,truth,ids = scene(rot)
        hit = ids>0
        band = ((ids==1)&dilate(dilate(ids==2))) | ((ids==2)&dilate(dilate(ids==1)))
        clear = hit & ~band & ~dilate(dilate(~hit))
        colors,depths = [],[]
        rays = [r.reshape(-1,r.shape[-1]).cuda() for r in rays]
        for name,model in zip(('uniform','canny'),models):
            chunks = [model(*b,return_depth=True)[1:] for b in zip(*(r.split(1024) for r in rays))]
            pred,z = [torch.cat(q).cpu().numpy() for q in zip(*chunks)]
            pred,z = pred.reshape(H,W,3),z.reshape(H,W)
            err = abs(z-truth)
            records.append((view,name,-10*np.log10(((pred[hit]-rgb[hit]/255)**2).mean()),
                            err[band].mean() if band.any() else np.nan,err[clear].mean(),int(band.sum())))
            colors.append(np.rint(pred.clip(0,1)*255).astype('uint8'))
            depths.append(z)
        if view=='demo':
            panels(list(zip(('Reference','Uniform depth loss','Canny-masked depth loss'),[rgb,*colors])),dest/'rgb.png')
            line=H//2
            pairs=(ids[line,:-1]==1)&(ids[line,1:]==2)
            edge=int(np.flatnonzero(pairs)[0])
            zs=[truth,*depths]
            def depth_image(z):
                value=(np.where(hit,z,5).clip(2.5,5)-2.5)/2.5
                image=np.repeat(np.rint(255*(1-value)).astype('uint8')[:,:,None],3,axis=2)
                image[~hit]=[40,49,65]
                image[line,edge-20:edge+22:3]=[255,100,30]
                return image
            panels([(label,depth_image(z)) for label,z in zip(('Reference depth','Uniform depth','Canny-masked depth'),zs)],dest/'depth.png')
            xs=np.arange(edge-20,edge+22)
            chart=Image.new('RGB',(760,300),'white'); pen=ImageDraw.Draw(chart)
            x0,x1,y0,y1=70,730,55,240
            def yy(v): return int(y1-(v-2.5)/2.5*(y1-y0))
            def xx(x): return x0+(x-xs[0])*(x1-x0)/(xs[-1]-xs[0])
            for val in [2.5,3.,3.5,4.,4.5,5.]:
                pen.line([(x0,yy(val)),(x1,yy(val))],fill='#dedede')
                pen.text((35,yy(val)-6),str(val),fill='black')
            for x in [xs[0],edge,edge+1,xs[-1]]:
                pen.line([(xx(x),y1),(xx(x),y1+4)],fill='black')
                if x!=edge+1: pen.text((xx(x)-8,y1+8),str(x),fill='black')
            pen.text((215,275),f'Pixel column on row {line}   |   overlap between columns {edge} and {edge+1}',fill='black')
            for j,(label,z,col) in enumerate(zip(['Reference','Uniform','Canny'],zs,['black','#d66016','#176bb4'])):
                pen.line([(xx(x),yy(z[line,x])) for x in xs],fill=col,width=3)
                pen.text((190+j*115,15),label,fill=col)
                print(label,'adjacent depths',z[line,edge:edge+2],flush=True)
            chart.save(dest/'profile.png')
    with (out/'per_view.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['view','mode','foreground_psnr','boundary_mae','interior_mae','boundary_pixels']);writer.writerows(records)
    for mode in ('uniform','canny'):
        rows=[r for r in records if r[1]==mode and r[0]!='demo']
        print(mode,np.nanmean(np.array([r[2:5] for r in rows]),axis=0),'views with overlap',sum(r[5]>0 for r in rows),flush=True)

if __name__ == '__main__':
    main()
