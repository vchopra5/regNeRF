"""Test the partner's committed L1 depth loss on scan 40."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REVISION = '92d882b6f5106c78fccb2c67691b6eb80652a358'
BASE = Path('checkpoint/scan40_canny_pair_20261103')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args]).decode('utf-8', errors='replace').strip()


def prepare(output):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This run needs CUDA.')
    baseline = json.loads((BASE / 'protocol.json').read_text(encoding='utf-8'))
    assert baseline['steps'] == 20000 and baseline['seed'] == 42 and baseline['stratified']
    assert baseline['training_views'] == [25, 22, 28]
    for name, expected in baseline['training_input_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'The training input changed: {name}')
    checkpoint = BASE / 'regnerf/checkpoint_20000.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['step'] == 20000 and saved['model']['_extra_state']['stratified']
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    for name in ('train.py', 'model.py', 'data.py', 'dtu.py'):
        content = subprocess.check_output(['git', 'show', f'{REVISION}:{name}'])
        (source / name).write_bytes(content)
        if name != 'train.py':
            assert (source / name).read_text(encoding='utf-8') == (BASE / 'source' / name).read_text(encoding='utf-8')
    for name in ('eval_dtu.py', 'render.py'):
        shutil.copy2(BASE / 'source' / name, source / name)
    shutil.copy2(__file__, source / 'compare_partner_dtu.py')
    record = dict(partner_revision=REVISION, working_revision=git('rev-parse', 'HEAD'),
                  local_status=git('status', '--short'), local_diff=git('diff', 'HEAD'),
                  source_sha256={p.name: sha(p) for p in source.glob('*.py')},
                  scene=str(Path(baseline['scene']).resolve()), seed=42, steps=20000,
                  training_views=[25, 22, 28], test_views=baseline['test_views'],
                  training_scale=4, evaluation_scale=2, stratified=True,
                  depth_weight=0.1, depth_weight_start=400.0, coarse_depth_weight=0.1,
                  edge_loss_weight=0.1, edge_scale_quantile=0.9, edge_halo=2,
                  selection='Fixed repository example weight; no test-view tuning',
                  adaptation='Move training images to CUDA for edge scale calculation; loss code unchanged',
                  optimizer='Adam', config=baseline['config'], bounds=baseline['bounds'],
                  depth_sampling_seed=43, background='black', density_noise_std=1.0,
                  regnerf_density=True, device=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, baseline_protocol=baseline,
                  baseline_checkpoint_sha256=sha(checkpoint),
                  checkpoints=dict(regnerf=str(checkpoint.resolve()),
                                   partner=str(output / 'partner/checkpoint_20000.pt')),
                  started_utc=datetime.now(timezone.utc).isoformat())
    write(output / 'protocol.json', record)


def worker(output):
    import torch
    import train as training
    assert Path(training.__file__).resolve() == output / 'source/train.py'
    record = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    training.PRINT_EVERY = 250
    for key, value in record['config'].items():
        assert getattr(training, key) == value, key
    original_load = training.load_data

    def load_on_gpu(*args, **kwargs):
        images, poses, intrinsics = original_load(*args, **kwargs)
        return images.cuda(), poses, intrinsics

    training.load_data = load_on_gpu
    start = time.monotonic()

    def progress(step, model):
        if step == 1 or step % 250 == 0:
            elapsed = time.monotonic() - start
            write(output / 'status.json', dict(state='training', mode='partner', step=step,
                  steps=20000, elapsed_seconds=elapsed, updated_utc=datetime.now(timezone.utc).isoformat()))
            print(f'Partner: {step}/20000; elapsed {elapsed:.0f} seconds', flush=True)

    try:
        training.train(3, Path(record['scene']), output / 'partner', depth_weight=0.1,
                       depth_weight_start=400.0, stop_after=20000, after_step=progress,
                       background='black', density_noise_std=1.0, regnerf_density=True,
                       scale=4, seed=42, stratified=True, edge_loss_weight=0.1)
        scores = {}
        for mode in ('regnerf', 'partner'):
            write(output / 'status.json', dict(state='evaluation', mode=mode))
            score_file = output / f'{mode}_scores.json'
            subprocess.run([sys.executable, '-u', str(output / 'source/eval_dtu.py'),
                            '--checkpoint', record['checkpoints'][mode], '--data-dir', record['scene'],
                            '--scale', '2', '--device', 'cuda', '--output-json', str(score_file)], check=True)
            scores[mode] = json.loads(score_file.read_text(encoding='utf-8'))
            assert set(scores[mode]['per_view_foreground_psnr']) == set(map(str, record['test_views']))
        means = {m: s['mean_foreground_psnr'] for m, s in scores.items()}
        summary = dict(means_db=means, partner_minus_regnerf_db=means['partner']-means['regnerf'],
                       scores=scores, steps=20000, evaluation_scale=2,
                       elapsed_seconds=time.monotonic()-start,
                       scope='Single-scan investigation of RegNeRF Table 3. The paper uses a different setup and a benchmark aggregate.',
                       geometry='No scan 40 reference depth is present. Boundary and flat-surface depth errors are not measured.')
        write(output / 'summary.json', summary)
        write(output / 'status.json', dict(state='complete', **summary))
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        write(output / 'status.json', dict(state='failed', error=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(output)
    elif args.run:
        prepare(output)
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(output / 'source/compare_partner_dtu.py'),
                            '--output-dir', str(output), '--worker'], stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        print('Use --run to train the partner model for 20000 steps, then score it and the saved baseline.')
        print('Use seed 42, views 25, 22, 28, jitter, scale 4, test scale 2, and L1 weight 0.1.')


if __name__ == '__main__':
    main()
