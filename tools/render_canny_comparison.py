"""Render matched RGB and depth views from the saved scan 40 pair."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PAIR = Path('checkpoint/scan40_canny_pair_20261103')
sys.path.insert(0, str((PAIR / 'source').resolve()))
import dtu
from data import get_rays
from model import load_model, render_rays

OUTPUT = Path('renders/scan40_canny_pair_20261103')
VIEWS = (10, 24, 43)
MODES = ('regnerf', 'canny_weight')


@torch.inference_mode()
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene = Path('data/dtu/scan40')
    scores = {m: json.loads((PAIR / f'{m}_scores.json').read_text()) for m in MODES}
    models = {m: load_model(PAIR / m / 'checkpoint_20000.pt', device,
                           near=dtu.BOUNDS[0], far=dtu.BOUNDS[1]) for m in MODES}
    results = {}
    with np.load(scene / 'cameras.npz') as cameras:
        for view in VIEWS:
            assert view in dtu.TEST_IDS
            with Image.open(scene / 'image' / f'{view:06}.png') as image:
                h, w, intrinsics, pose = dtu.camera(cameras, view, image.size, 2)
                reference = image.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
            with Image.open(scene / 'mask' / f'{view:03}.png') as image:
                mask = np.asarray(image.convert('L').resize((w, h), Image.Resampling.NEAREST)) > 127
            reference.save(OUTPUT / f'view{view}_reference.png')
            results[view] = dict(reference=reference, mask=mask)
            rays = [r.reshape(h*w, -1) for r in get_rays(h, w, intrinsics, pose)]
            for mode, model in models.items():
                rgb, depth = render_rays(model, rays, 1024, device, return_depth=True)
                rgb = rgb.cpu().numpy().reshape(h, w, 3)
                depth = depth.cpu().numpy().reshape(h, w)
                assert np.isfinite(rgb).all() and np.isfinite(depth).all()
                np.savez_compressed(OUTPUT / f'view{view}_{mode}.npz', rgb=rgb, depth=depth, mask=mask)
                picture = Image.fromarray(np.rint(rgb.clip(0, 1)*255).astype(np.uint8))
                picture.save(OUTPUT / f'view{view}_{mode}_rgb.png')
                results[view][mode] = (picture, depth)
                print(f'Rendered view {view}: {mode}', flush=True)
    values = np.concatenate([results[v][m][1][results[v]['mask']] for v in VIEWS for m in MODES])
    low, high = np.percentile(values, (2, 98)).tolist()
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    title = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    for view, result in results.items():
        board = Image.new('RGB', (1240, 790), '#faf9f5')
        draw = ImageDraw.Draw(board)
        draw.text((16, 10), f'Stone arch | test view {view} | 20,000 steps | seed 42', font=title, fill='#172d37')
        draw.text((16, 48), 'One shared depth scale for both models and all three views.', font=font, fill='#334955')
        labels = ('Reference photo', 'Standard RegNeRF', 'Canny-based weights')
        for i, label in enumerate(labels):
            x = 10 + 410*i
            draw.text((x, 82), label, font=font, fill='#172d37')
            picture = result['reference'] if i == 0 else result[MODES[i-1]][0]
            board.paste(picture.resize((400, 300), Image.Resampling.LANCZOS), (x, 112))
            if i:
                mode = MODES[i-1]
                score = scores[mode]['per_view_foreground_psnr'][str(view)]
                draw.text((x, 420), f'Foreground RGB PSNR: {score:.2f} dB', font=small, fill='#334955')
                depth = result[mode][1]
                shade = np.rint(np.clip((depth-low)/(high-low), 0, 1)*255).astype(np.uint8)
                depth_rgb = np.repeat(shade[..., None], 3, axis=-1)
                depth_rgb[~result['mask']] = [218, 227, 235]
                image = Image.fromarray(depth_rgb)
                image.save(OUTPUT / f'view{view}_{mode}_depth.png')
                board.paste(image.resize((400, 300), Image.Resampling.NEAREST), (x, 455))
        draw.multiline_text((16, 470), 'Predicted depth\n\nDark = near; light = far.\nBlue-grey = outside the test mask.\n\nNo reference depth is available.\nA cleaner map can still be wrong.', font=font, fill='#172d37', spacing=9)
        draw.text((16, 727), f'Shared range: {low:.3f} to {high:.3f}', font=font, fill='#172d37')
        draw.text((16, 764), 'Camera-axis depth in scene units. Range clips the pooled foreground tails at 2% and 98%.', font=small, fill='#334955')
        board.save(OUTPUT / f'view{view}_comparison.png')
    (OUTPUT / 'render_record.json').write_text(json.dumps(dict(
        views=VIEWS, scale=2, steps=20000, depth_range=[low, high],
        selection='Diagnostic views chosen after scoring: view 10 has the largest RGB loss; view 24 improves; view 43 is another angle.',
        depth_display='Shared pooled foreground 2nd and 98th percentiles. Common test mask used only for display.',
        depth_units='Camera-axis depth in normalized scene units',
        source=str(PAIR / 'source'), device=device), indent=2), encoding='utf-8')
    print(f'Saved comparisons to {OUTPUT.resolve()}', flush=True)


if __name__ == '__main__':
    main()
