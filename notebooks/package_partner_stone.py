"""Package saved scan 40 evidence for Notebook 3. Do not train or render."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from notebooks.lab import ROOT
from notebooks.toy_figures import Board, TITLE, SMALL


def main():
    run = ROOT / 'checkpoint/scan40_partner_20261103'
    renders = ROOT / 'renders/scan40_partner_20261103'
    output = ROOT / 'notebooks/results/partner_stone'
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    render_record = json.loads((renders / 'render_record.json').read_text(encoding='utf-8'))
    modes = ('regnerf', 'partner')
    per_view = []
    for view in protocol['test_views']:
        values = {m: summary['scores'][m]['per_view_foreground_psnr'][str(view)] for m in modes}
        per_view.append(dict(view=view, regnerf_db=values['regnerf'], partner_db=values['partner'],
                             partner_minus_regnerf_db=values['partner']-values['regnerf']))
    assert len(per_view) == 25
    for mode in modes:
        assert abs(np.mean([r[f'{mode}_db'] for r in per_view])-summary['means_db'][mode]) < 1e-12
    low, high = render_record['depth_range']
    assert set(render_record['views']) == set(protocol['test_views'])
    for view in protocol['test_views']:
        with Image.open(renders / f'view{view}_reference.png') as image:
            reference = image.convert('RGB')
        for kind in ('rgb', 'depth'):
            board = Board(1206, 390)
            pictures = [reference]
            for mode in modes:
                with np.load(renders / f'view{view}_{mode}.npz') as arrays:
                    values = arrays[kind]
                    if kind == 'rgb':
                        pixels = np.rint(values.clip(0, 1)*255).astype(np.uint8)
                    else:
                        pixels = np.rint(np.clip((values-low)/(high-low), 0, 1)*255).astype(np.uint8)
                pictures.append(Image.fromarray(pixels).convert('RGB'))
            for i, (label, picture) in enumerate(zip(
                    ('Reference photo', 'RegNeRF', 'RegNeRF + edge L1'), pictures)):
                x = 8 + i*400
                board.text((x, 8), label, size=TITLE)
                board.paste(picture, (x, 38), (390, 292))
            if kind == 'depth':
                captions = [f'Depth: dark = {low:.3f}; light = {high:.3f} scene units. One scale for all views and both models.',
                            'Left: colour photo, not reference depth. No depth mask. Display clips pooled foreground tails at 2% and 98%.']
            else:
                row = next(r for r in per_view if r['view'] == view)
                captions = [f"Test view {view}: foreground PSNR, RegNeRF {row['regnerf_db']:.2f} dB; + edge L1 {row['partner_db']:.2f} dB.",
                            'Same camera, 20,000 steps, and seed 42. The score uses the full evaluation image at scale 2.']
            for j, caption in enumerate(captions):
                board.text((8, 344+j*20), caption, size=SMALL)
            board.image.save(output / f'view{view}_{kind}.png')
    checkpoints = {m: str(Path(p).resolve().relative_to(ROOT)) for m, p in protocol['checkpoints'].items()}
    record = dict(means_db=summary['means_db'], difference_db=summary['partner_minus_regnerf_db'],
                  per_view=per_view, wins=sum(r['partner_minus_regnerf_db'] > 0 for r in per_view),
                  settings={k: protocol[k] for k in ('seed', 'steps', 'training_views', 'test_views',
                      'training_scale', 'evaluation_scale', 'stratified', 'depth_weight', 'depth_weight_start',
                      'coarse_depth_weight', 'edge_loss_weight', 'edge_scale_quantile', 'edge_halo',
                      'selection', 'adaptation', 'config', 'device')},
                  checkpoints=checkpoints,
                  checkpoint_sha256={m: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for m,p in checkpoints.items()},
                  partner_revision=protocol['partner_revision'], source_sha256=protocol['source_sha256'],
                  baseline_revision=protocol['baseline_protocol']['code_revision'],
                  baseline_source_sha256=protocol['baseline_protocol']['source_sha256'],
                  training_input_sha256=protocol['baseline_protocol']['training_input_sha256'],
                  render_record=render_record, scope=summary['scope'], geometry=summary['geometry'],
                  source_record=str((run / 'protocol.json').relative_to(ROOT)))
    (output / 'results.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Saved 50 panels and 25-view scores in {output}')


if __name__ == '__main__':
    main()
