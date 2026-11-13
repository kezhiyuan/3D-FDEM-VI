from pathlib import Path
import argparse, hashlib, json, math, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text('utf8'))

def write(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu-granted', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from validate_protocol import require_frozen_protocol
    protocol = require_frozen_protocol()
    if not (args.execute and args.gpu_granted):
        raise ValueError('Explicit new all-data GPU allocation required')
    import torch
    import AEM_Variational_Bayes_3D_FD_v8 as core
    from spatial_coordinates import CosineSpatialCoordinates
    values = dict(protocol['config'])
    values.update(initial_model_path='', result_dir=str(Path(protocol['output']) / 'cold_DCT_MAP180'))
    cfg = core.Config(**values)
    dest = Path(cfg.result_dir)
    if dest.exists() and (not args.resume):
        raise FileExistsError('Preserve existing cold MAP')
    dest.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    core.configure_runtime(cfg)
    device = torch.device('cuda')
    problem = core.Problem(cfg, device)
    assert problem.n_stations == 3949
    options = protocol['map']
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    rng = np.random.default_rng(options['batch_seed'])
    coordinates = [CosineSpatialCoordinates((cfg.ny, cfg.nx, cfg.nz), v, options['epsilon'], device=device) for v in options['lengths_yxz']]
    uniform = (2.0 - cfg.model_min) / (cfg.model_max - cfg.model_min)
    z = torch.nn.Parameter(torch.full((1, 1, cfg.ny, cfg.nx, cfg.nz), math.log(uniform / (1 - uniform)), device=device))
    stage = 0
    optimizer = torch.optim.Adam([z], lr=options['learning_rate'])
    history = []
    transfers = []
    best = None
    selected = None
    first = 0
    offset = 0.0
    sequence = []
    protocol_sha = sha(HERE / 'execution_protocol.json')
    if args.resume:
        state = torch.load(dest / 'last.pt', map_location=device, weights_only=False)
        assert state['protocol_sha256'] == protocol_sha
        stage = state['stage']
        z.data.copy_(state['z'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng_numpy']
        torch.set_rng_state(state['rng_torch'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])
        history = state['history']
        transfers = state['transfers']
        best = state['best']
        selected = state['selected']
        first = state['step'] + 1
        offset = state['elapsed_seconds']
        sequence = state['sequence']

    def physical():
        return cfg.model_min + (cfg.model_max - cfg.model_min) * torch.sigmoid(coordinates[stage].apply(z))

    def evaluate():
        with torch.no_grad():
            m = physical()
            prediction = problem.forward_models(m, np.arange(3949))[0]
            residual = prediction + problem.bias - problem.label
            kernel = float((0.5 * (residual / problem.sigma_eff).square()).sum())
            normalizer = float((torch.log(problem.sigma_eff) + 0.5 * math.log(2 * math.pi)).double().sum())
            prior = float(core.prior_energy(m, cfg)[0])
        return ({'point_objective': cfg.data_weight * (kernel + normalizer) + prior, 'weighted_kernel': cfg.data_weight * kernel, 'normalizer_unweighted': normalizer, 'prior': prior, 'raw_RMS_ppm': float(residual.double().square().mean().sqrt())}, prediction)

    def checkpoint(step, path):
        torch.save({'step': step, 'stage': stage, 'z': z.detach().cpu(), 'optimizer': optimizer.state_dict(), 'rng_numpy': rng.bit_generator.state, 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(), 'config': values, 'protocol_sha256': protocol_sha, 'history': history, 'transfers': transfers, 'best': best, 'selected': selected, 'sequence': sequence, 'elapsed_seconds': offset + time.perf_counter() - clock}, path)
    for step in range(first, 181):
        if step:
            next_stage = min(2, (step - 1) // 60)
            if next_stage != stage:
                before = history[-1]['metrics']
                before_logits = coordinates[stage].apply(z).detach()
                before_model = physical().detach()
                with torch.no_grad():
                    z.copy_(coordinates[stage].rebase(z, coordinates[next_stage]))
                stage = next_stage
                after_logits = coordinates[stage].apply(z).detach()
                after_model = physical().detach()
                max_model = float((before_model - after_model).abs().max())
                max_logits = float((before_logits - after_logits).abs().max())
                assert max_model < 5e-05 and max_logits < 3e-05
                optimizer = torch.optim.Adam([z], lr=options['learning_rate'])
                (after, _) = evaluate()
                delta = (after['point_objective'] - before['point_objective']) / max(1.0, abs(before['point_objective']))
                assert abs(delta) < 0.0002
                transfers.append({'after_step': step - 1, 'max_logit_change': max_logits, 'max_log10rho_change': max_model, 'before': before, 'after': after, 'relative_complete_objective_change': delta})
                write(dest / 'coordinate_transfers.json', transfers)
                del before_logits, before_model, after_logits, after_model
            local = (step - 1) % 60 + 1
            optimizer.param_groups[0]['lr'] = options['learning_rate'] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * local / 60)))
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
                print('[Cold all MAP]', step, 'stage', stage, 'working_RMS', rms, flush=True)
        if step in options['training_evaluation_steps']:
            (score, prediction) = evaluate()
            history.append({'step': step, 'metrics': score, 'elapsed_seconds': offset + time.perf_counter() - clock})
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
    report = {'status': 'completed', 'station_count': 3949, 'cold_uniform100': True, 'truth_or_GN_used': False, 'selected_step': selected, 'selected_training_metrics': best, 'selected_model_sha256': sha(dest / 'selected_model_log10.npy'), 'selected_checkpoint_sha256': sha(dest / 'best.pt'), 'protocol_sha256': protocol_sha, 'data_sha256': sha(cfg.data_path), 'index_sha256': sha(cfg.index_path), 'elapsed_seconds_including_load': offset + time.perf_counter() - clock, 'source_sha256': sha(__file__), 'not_a_posterior': True}
    write(dest / 'completed_MAP.json', report)
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
