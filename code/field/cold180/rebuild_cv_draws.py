from __future__ import annotations
import argparse
import base64
import hashlib
import io
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recipe', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu-granted', action='store_true')
    p.add_argument('--restore-subset', type=Path)
    args = p.parse_args()
    recipe = json.loads(args.recipe.read_text('utf-8'))
    here = Path(recipe['core']).parent
    for (name, expected) in recipe['source_sha256'].items():
        if sha(here / name) != expected:
            raise ValueError('Archived source changed: ' + name)
    if sha(recipe['checkpoint']) != recipe['checkpoint_sha256']:
        raise ValueError('Archived checkpoint changed')
    print(json.dumps({'CPU_binding_checks': 'pass', 'draws': recipe['draws'], 'subset': recipe['quantile_event_subset'], 'execute': args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if not args.gpu_granted:
        p.error('Requires exclusive GPU assignment')
    if args.restore_subset and args.restore_subset.exists():
        raise FileExistsError('Do not overwrite a preserved sample cube')
    import torch
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location('_cv_replay_frozen_core', recipe['core'])
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    configure_runtime = core.configure_runtime
    config_from_checkpoint = core.config_from_checkpoint
    MultiscaleFlow3D = core.MultiscaleFlow3D
    payload = torch.load(recipe['checkpoint'], map_location='cpu', weights_only=False)
    cfg = config_from_checkpoint(payload)
    configure_runtime(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow = MultiscaleFlow3D(cfg, device).to(device)
    flow.load_state_dict(payload['flow'])
    flow.set_active_coupling_layers(len(flow.layers))
    flow.eval()
    del payload
    generator = torch.Generator(device=device)
    state = np.frombuffer(base64.b64decode(recipe['initial_generator_state_base64']), dtype=np.uint8).copy()
    generator.set_state(torch.as_tensor(state, dtype=torch.uint8))
    n = recipe['draws']
    subset_n = recipe['quantile_event_subset']
    shape = tuple(recipe['model_shape'])
    stream_hash = hashlib.sha256()
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': np.lib.format.dtype_to_descr(np.dtype('float32')), 'fortran_order': False, 'shape': (subset_n, *shape)})
    subset_hash = hashlib.sha256(header.getvalue())
    restored = None
    if args.restore_subset:
        args.restore_subset.parent.mkdir(parents=True, exist_ok=True)
        restored = np.lib.format.open_memmap(args.restore_subset, mode='w+', dtype='float32', shape=(subset_n, *shape))
    sums = {key: torch.zeros(shape, device=device, dtype=torch.float64) for key in ('rho', 'rho_square', 'm', 'm_square')}
    moment_checks = {}
    tick = time.perf_counter()
    status = 'failed'
    error = None
    try:
        with torch.no_grad():
            for first in range(0, n, recipe['batch_size']):
                (m, _) = flow.sample_and_log_prob(min(recipe['batch_size'], n - first), device, antithetic=False, generator=generator)
                value = m[:, 0].double()
                rho = torch.exp(value * math.log(10.0))
                sums['rho'] += rho.sum(0)
                sums['rho_square'] += rho.square().sum(0)
                sums['m'] += value.sum(0)
                sums['m_square'] += value.square().sum(0)
                array = m[:, 0].cpu().numpy().astype('float32', copy=False)
                stream_hash.update(array.tobytes(order='C'))
                if first < subset_n:
                    count = min(len(array), subset_n - first)
                    subset_hash.update(array[:count].tobytes(order='C'))
                    if restored is not None:
                        restored[first:first + count] = array[:count]
                if first == 0 or (first + len(array)) % 128 == 0:
                    print('[Replay]', first + len(array), '/', n, flush=True)
        if stream_hash.hexdigest() != recipe['all_draw_payload_sha256']:
            raise ValueError('Actual1024 draw payload differs from archive')
        if subset_hash.hexdigest() != recipe['subset_npy_sha256']:
            raise ValueError('Actual first256 NPY stream differs from archive')
        replayed = {}
        for (suffix, sum_key, square_key) in [('rho', 'rho', 'rho_square'), ('log10', 'm', 'm_square')]:
            mean = sums[sum_key] / n
            variance = ((sums[square_key] - sums[sum_key].square() / n) / (n - 1)).clamp_min(0.0)
            replayed['mean_' + suffix] = mean.cpu().numpy().astype(np.float32)
            replayed['std_' + suffix] = variance.sqrt().cpu().numpy().astype(np.float32)
        replayed['relative_std_rho'] = replayed['std_rho'] / replayed['mean_rho']
        for (name, value) in replayed.items():
            stored_path = args.recipe.parent / (name + '.npy')
            if not stored_path.exists():
                raise FileNotFoundError('Archived physical/log moment missing: ' + name)
            stored = np.load(stored_path)
            error_array = np.abs(value.astype(np.float64) - stored.astype(np.float64))
            moment_checks[name] = {'stored_sha256': sha(stored_path), 'exact_array_equal': bool(np.array_equal(value, stored)), 'max_abs_error': float(error_array.max()), 'allclose_rtol5e_7_atol1e_6': bool(np.allclose(value, stored, rtol=5e-07, atol=1e-06))}
            if not moment_checks[name]['allclose_rtol5e_7_atol1e_6']:
                raise ValueError('Replayed physical/log moment differs: ' + name)
        if restored is not None:
            restored.flush()
            restored._mmap.close()
            restored = None
            if sha(args.restore_subset) != recipe['subset_npy_sha256']:
                raise ValueError('Restored subset file differs from archive')
        status = 'pass'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if restored is not None:
            restored.flush()
            restored._mmap.close()
        report = {'schema': 'actual_cv_v3_hash_replay', 'status': status, 'error': error, 'recipe': str(args.recipe.resolve()), 'recipe_sha256': sha(args.recipe), 'actual_executed_core': str(Path(core.__file__).resolve()), 'actual_executed_core_sha256': sha(Path(core.__file__)), 'replayer_source_sha256': sha(Path(__file__)), 'checkpoint_sha256': recipe['checkpoint_sha256'], 'draws_actually_generated': n if status == 'pass' else None, 'replayed_payload_sha256': stream_hash.hexdigest(), 'expected_payload_sha256': recipe['all_draw_payload_sha256'], 'replayed_subset_npy_sha256': subset_hash.hexdigest(), 'expected_subset_npy_sha256': recipe['subset_npy_sha256'], 'independently_recomputed_model_moments': moment_checks, 'seconds': time.perf_counter() - tick, 'restore_subset': str(args.restore_subset) if args.restore_subset else None, 'mode': 'actual full1024 replay; hash only unless subset output explicitly requested'}
        output = args.recipe.parent / ('actual_replay_' + str(time.time_ns()) + '.json')
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
