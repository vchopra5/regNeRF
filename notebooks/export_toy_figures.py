"""Build static figures from the same saved arrays used in the toy lesson."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from notebooks.lab import solve_depth
from notebooks.toy_experiment import ASSETS, ensure_assets
from notebooks.toy_figures import (comparison, crisp, curve_fit_figure, depth_profile, font,
                                  patch_experiment, shear_patch, surface_inspector)


def stack(panels, width=1200):
    resized = []
    for title, picture in panels:
        picture = picture.convert('RGB')
        picture = picture.resize((width, round(picture.height * width / picture.width)), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (width, picture.height + 52), 'white')
        ImageDraw.Draw(panel).text((16, 12), title, fill='#203743', font=font(28))
        panel.paste(picture, (0, 52))
        resized.append(panel)
    board = Image.new('RGB', (width, sum(p.height for p in resized)), 'white')
    top = 0
    for panel in resized:
        board.paste(panel, (0, top))
        top += panel.height
    return board


def main():
    ensure_assets()
    out = Path(__file__).resolve().parent / 'toy_figures'
    out.mkdir(exist_ok=True)
    with np.load(ASSETS / 'reference.npz') as data:
        truth, ids = data['depth'], data['ids']
    with np.load(ASSETS / 'coverage_500/3/prediction.npz') as data:
        baseline = data['depth']
    angle = json.loads((ASSETS / 'transforms_train.json').read_text())['camera_angle_x']
    focal = truth.shape[1] / (2*np.tan(angle/2))
    rgb = np.asarray(Image.open(ASSETS / 'reference.png'))
    coverage = []
    for count in (3, 9, 20):
        folder = ASSETS / 'coverage_500' / str(count)
        score = json.loads((folder / 'result.json').read_text())
        inputs = Image.open(folder / 'inputs.png').convert('RGB')
        inputs = inputs.resize((1200, round(inputs.height * 1200 / inputs.width)), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (1200, 520 + inputs.height), 'white')
        panel.paste(crisp(ASSETS / 'reference.png', 300), (0, 0))
        panel.paste(crisp(folder / 'heldout.png', 300), (620, 0))
        ImageDraw.Draw(panel).text((16, 456), 'Reference', fill='#203743', font=font(28))
        ImageDraw.Draw(panel).text((636, 456), f'{count} photos: {score["foreground_psnr"]:.2f} dB',
                                   fill='#203743', font=font(28))
        panel.paste(inputs, (0, 520))
        coverage.append((f'{count} photos; 500 steps; smoothing off', panel))
    stack(coverage).save(out / 'coverage_500.png')
    stack([(f'Camera shift {s:+.1f}', shear_patch(rgb, truth, baseline, focal, s)) for s in (0,.8)]).save(out / 'camera_shift.png')
    shapes = []
    for name,(left,right) in {'Centre':(55,75),'Near the rim':(90,110),'Full width':(25,110)}.items():
        a,b=truth[75,left:right+1],baseline[75,left:right+1]
        limits=(min(a.min(),b.min())-.03,max(a.max(),b.max())+.03)
        surface=surface_inspector(truth,baseline,ids,focal,25,75,left,right)
        profile=depth_profile(truth,baseline,ids==1,75,left,right,limits=limits)
        panel=stack([('Same red sphere; saved three-photo baseline',surface),('Depth along orange strip',profile)])
        shapes.append((name,panel))
    stack(shapes).save(out/'shape_presets.png')
    true=truth[75,27:109]
    readings=true+np.random.default_rng(42).normal(0,.06,len(true))
    stack([(f'Simulated one-dimensional fit; smoothing {w}',curve_fit_figure(true,readings,solve_depth(readings,w))) for w in (0,4,256)]).save(out/'curve_settings.png')
    for whole,name in [(False,'one_pixel.png'),(True,'whole_patch.png')]:
        patch_experiment(truth[71:79,62:70],.1,whole)[0].save(out/name)
    training=[]
    for camera_name, camera_scene in [('Lower view',ASSETS/'lower'),('Front view',ASSETS)]:
        for steps in (100,300,1000):
            off=camera_scene/('off' if steps==300 else f'off_{steps}')
            on=camera_scene/('on' if steps==300 else f'on_{steps}')
            panel=comparison(camera_scene,[('Smoothing off',off),('Smoothing on',on)])
            a,b=[json.loads((p/'result.json').read_text()) for p in (off,on)]
            delta=b['foreground_psnr']-a['foreground_psnr']
            training.append((f'{camera_name}; {steps:,} steps; RGB PSNR change {delta:+.2f} dB',panel))
    stack(training).save(out/'training_stages.png')
    print(f'Saved fallback figures to {out}')


if __name__=='__main__':
    main()
