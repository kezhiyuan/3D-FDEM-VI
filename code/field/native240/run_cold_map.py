from pathlib import Path
import argparse, hashlib, json, math, time, os
import numpy as np
HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda : stream.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text('utf8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu-granted', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from validate_protocol import require_frozen_protocol
    p = require_frozen_protocol()
    if not (args.execute and args.gpu_granted):
        raise ValueError('New explicit root GPU allocation required')
    import torch
    import AEM_Variational_Bayes_3D_FD_v8 as core
    options = p['native_refinement']
    source = Path(options['source_cold_MAP_checkpoint'])
    source_meta = read(options['source_cold_MAP_completion'])
    assert sha(source) == source_meta['selected_checkpoint_sha256'] == options['source_cold_MAP_checkpoint_sha256']
    assert source_meta['status'] == 'completed' and source_meta['cold_uniform100'] and (not source_meta['truth_or_GN_used'])
    assert source_meta['selected_step'] == 180 and source_meta['station_count'] == 3949
    origin = read(options['source_cold_protocol'])
    assert source_meta['protocol_sha256'] == sha(options['source_cold_protocol'])
    values = dict(p['config'])
    values.update(initial_model_path='', result_dir=str(Path(p['output']) / 'native_MAP240'))
    cfg = core.Config(**values)
    dest = Path(cfg.result_dir)
    if dest.exists() and (not args.resume):
        raise FileExistsError('Preserve any prior refinement')
    dest.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    core.configure_runtime(cfg)
    device = torch.device('cuda')
    problem = core.Problem(cfg, device)
    original = torch.load(source, map_location='cpu', weights_only=False)
    assert original['step'] == 180 and original['stage'] == 2
    for (key, value) in origin['config'].items():
        if key not in ('result_dir', 'initial_model_path'):
            assert values[key] == value, key
    z = torch.nn.Parameter(original['z'].to(device).clone())
    source_model = np.load(options['source_cold_MAP_model'])
    m0 = (cfg.model_min + (cfg.model_max - cfg.model_min) * torch.sigmoid(z)).detach()[0, 0].cpu().numpy()
    source_max_error = float(np.max(np.abs(m0.astype(float) - source_model.astype(float))))
    assert source_max_error < 5e-05
    rng = np.random.default_rng()
    rng.bit_generator.state = original['rng_numpy']
    torch.set_rng_state(original['rng_torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in original['rng_cuda']])
    optimizer = torch.optim.Adam([z], lr=options['learning_rate'])
    history = []
    sequence = []
    best = None
    selected = None
    first = 0
    offset = 0.0
    if args.resume:
        state = torch.load(dest / 'last.pt', map_location=device, weights_only=False)
        assert state['protocol_sha256'] == sha(HERE / 'execution_protocol.json')
        z.data.copy_(state['z'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng_numpy']
        torch.set_rng_state(state['rng_torch'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])
        history = state['history']
        sequence = state['sequence']
        best = state['best']
        selected = state['selected']
        first = state['step'] + 1
        offset = state['elapsed_seconds']
    del original

    def physical():
        return cfg.model_min + (cfg.model_max - cfg.model_min) * torch.sigmoid(z)

    def evaluate():
        with torch.no_grad():
            m = physical()
            prediction = problem.forward_models(m, np.arange(3949))[0]
            residual = prediction + problem.bias - problem.label
            kernel = float((0.5 * (residual / problem.sigma_eff).square()).sum())
            constant = float((torch.log(problem.sigma_eff) + 0.5 * math.log(2 * math.pi)).double().sum())
            prior = float(core.prior_energy(m, cfg)[0])
        return ({'point_objective': cfg.data_weight * (kernel + constant) + prior, 'weighted_kernel': cfg.data_weight * kernel, 'normalizer_unweighted': constant, 'prior': prior, 'raw_RMS_ppm': float(residual.double().square().mean().sqrt())}, prediction)

    def checkpoint(step, path):
        temp = path.with_suffix('.tmp')
        torch.save({'step': step, 'stage': 2, 'coordinates': 'native_identity', 'z': z.detach().cpu(), 'optimizer': optimizer.state_dict(), 'rng_numpy': rng.bit_generator.state, 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(), 'config': values, 'protocol_sha256': sha(HERE / 'execution_protocol.json'), 'history': history, 'sequence': sequence, 'best': best, 'selected': selected, 'elapsed_seconds': offset + time.perf_counter() - clock, 'source_cold_MAP_checkpoint_sha256': sha(source)}, temp)
        os.replace(temp, path)
    for step in range(first, options['steps'] + 1):
        if step:
            lr = options['learning_rate'] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / options['steps'])))
            optimizer.param_groups[0]['lr'] = lr
            ids = rng.choice(3949, 66, replace=False)
            sequence.append(ids.tolist())
            optimizer.zero_grad(set_to_none=True)
            m = physical()
            (phi, gradient, rms, _) = core.streamed_gradient(m, ids, np.arange(5), problem, 1.0, 39490)
            prior = core.prior_energy(m, cfg).mean() / 39490
            torch.autograd.backward((m, prior), (gradient, None))
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            del m, phi, gradient, prior
            if step % 10 == 0:
                print('[Native structure refinement]', step, '/', options['steps'], 'working_RMS', rms, flush=True)
        if step in options['training_evaluation_steps']:
            (score, prediction) = evaluate()
            history.append({'step': step, 'metrics': score, 'elapsed_seconds': offset + time.perf_counter() - clock})
            if step == 0:
                relative_error = (score['point_objective'] - source_meta['selected_training_metrics']['point_objective']) / source_meta['selected_training_metrics']['point_objective']
                assert abs(relative_error) < 0.0002
                write(dest / 'source_point_reconstruction.json', {'status': 'pass', 'source_model_max_error_log10rho': source_max_error, 'source_complete_objective_relative_error': relative_error, 'source_checkpoint_sha256': sha(source), 'source_model_sha256': sha(options['source_cold_MAP_model']), 'prior_q_or_GN_used': False})
            if best is None or score['point_objective'] < best['point_objective']:
                best = score
                selected = step
                with torch.no_grad():
                    np.save(dest / 'selected_model_log10.npy', physical()[0, 0].cpu().numpy())
                np.savez_compressed(dest / 'selected_responses.npz', prediction_canonical=prediction.cpu().numpy(), observation_canonical=problem.label.cpu().numpy(), source_row_ids=np.arange(3949))
                checkpoint(step, dest / 'best.pt')
            write(dest / 'training_history.json', history)
        if step % 20 == 0:
            checkpoint(step, dest / 'last.pt')
    actual = offset + time.perf_counter() - clock
    report = {'status': 'completed', 'station_count': 3949, 'cold_uniform100': True, 'cold_uniform100_is_verified_origin_not_this_local_phase': True, 'truth_or_GN_used': False, 'source_cold_MAP_steps': 180, 'additional_native_MAP_steps': options['steps'], 'selected_step': selected, 'selected_total_point_step': 180 + selected, 'selected_training_metrics': best, 'selected_model_sha256': sha(dest / 'selected_model_log10.npy'), 'selected_checkpoint_sha256': sha(dest / 'best.pt'), 'protocol_sha256': sha(HERE / 'execution_protocol.json'), 'data_sha256': sha(cfg.data_path), 'index_sha256': sha(cfg.index_path), 'elapsed_seconds_including_load': actual, 'inherited_cold_MAP_seconds': source_meta['elapsed_seconds_including_load'], 'cumulative_point_structure_seconds': actual + source_meta['elapsed_seconds_including_load'], 'source_sha256': sha(__file__), 'source_cold_MAP_checkpoint_sha256': sha(source), 'not_a_posterior': True, 'selection_uses_complete_training_J_not_raw_RMS': True, 'separate_reserved_prediction_or_truth_files_used_for_selection': False, 'all3949_contains_historically_reserved_stations': True, 'new_independent_holdout_claim': False}
    write(dest / 'completed_MAP.json', report)
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
