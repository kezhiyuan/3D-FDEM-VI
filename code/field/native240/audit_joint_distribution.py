from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
HERE = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text('utf8'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omega', type=int, choices=[1, 50], default=50)
    parser.add_argument('--gpu-granted', action='store_true')
    args = parser.parse_args()
    if not args.gpu_granted:
        raise ValueError('Coordinator must allocate uniqueGPU')
    import torch
    import AEM_Variational_Bayes_3D_FD_v8 as e
    from validate_protocol import validate_training_inputs
    from cv_constraints import independent_moments, cv_penalty_from_moments, describe_cv
    assert args.omega == 50, 'Only the separately frozen original omega50 all-data target'
    protocol_path = HERE / 'execution_protocol.json'
    protocol = read(protocol_path)
    cfg = e.Config(**protocol['config'])
    validate_training_inputs(cfg)
    e.configure_runtime(cfg)
    dest = Path(cfg.result_dir) / 'independent_audit'
    if dest.exists():
        raise FileExistsError('Independent audit is frozen and not repeatedly sampled to pass')
    dest.mkdir()
    started = time.perf_counter()
    device = torch.device('cuda')
    problem = e.Problem(cfg, device)
    paths = {'initial': Path(cfg.result_dir) / 'initial_q.pt', 'selected': Path(cfg.result_dir) / 'best.pt'}
    payload = {key: torch.load(path, map_location='cpu', weights_only=False) for (key, path) in paths.items()}
    for p in payload.values():
        assert p['format'] == 'vi_v8_multiscale' and p['config'] == protocol['config']
    assert payload['initial']['step'] == 0
    changed = [k for (k, v) in payload['initial']['flow'].items() if not torch.equal(v, payload['selected']['flow'][k])]
    flow = e.MultiscaleFlow3D(cfg, device).eval()
    generator = torch.Generator(device=device).manual_seed(protocol['independent_J_audit_seed'])
    audit = []
    response_arrays = {}
    normalizer = float((torch.log(problem.sigma_eff) + 0.5 * math.log(2 * math.pi)).sum())
    with torch.no_grad():
        for k in range(4):
            z = torch.randn((1, *flow.base_shape), device=device, generator=generator)
            row = {}
            for key in ['initial', 'selected']:
                flow.load_state_dict(payload[key]['flow'])
                flow.set_active_coupling_layers(len(flow.layers))
                (model, logq) = flow.forward_from_base(z)
                response = problem.forward_models(model, np.arange(problem.n_stations))
                phi = float(e.likelihood(response, problem.label, problem.bias, problem.sigma_eff, problem.n_stations, cfg=cfg)[0])
                prior = float(e.prior_energy(model, cfg)[0])
                q = float(logq[0])
                J = phi + prior + q
                row[key] = {'weighted_Gaussian_kernel': phi, 'unweighted_Gaussian_log_sigma_constant': normalizer, 'prior': prior, 'logq': q, 'J_kernel': J, 'J_complete_Gaussian': J + cfg.data_weight * normalizer, 'raw_sample_RMS_ppm': float((response[0] + problem.bias - problem.label).square().mean().sqrt()), 'model_payload_sha256': hashlib.sha256(model[0, 0].cpu().numpy().tobytes()).hexdigest()}
                response_arrays[f'{key}_draw{k}'] = response[0].cpu().numpy()
            row['paired_delta_J_per_variable'] = (row['selected']['J_kernel'] - row['initial']['J_kernel']) / flow.numel
            audit.append(row)
            print('independent J pair', k + 1, 'delta/D', row['paired_delta_J_per_variable'], flush=True)
        moment_arrays = {}
        CV = {}
        for key in ['initial', 'selected']:
            flow.load_state_dict(payload[key]['flow'])
            flow.set_active_coupling_layers(len(flow.layers))
            gen = torch.Generator(device=device).manual_seed(protocol['independent_CV_audit_seed'])
            (mu, nu) = independent_moments(flow, device, 256, cfg.cv_batch_size, gen)
            (penalty, cv2, _) = cv_penalty_from_moments(mu, nu, 256, cfg.cv_target, cfg.cv_weight)
            cv = cv2.sqrt()[0, 0].cpu().numpy()
            moment_arrays[key + '_mean_rho'] = mu[0, 0].cpu().numpy()
            moment_arrays[key + '_second_moment_rho'] = nu[0, 0].cpu().numpy()
            moment_arrays[key + '_CV'] = cv
            CV[key] = {'draws': 256, 'seed': protocol['independent_CV_audit_seed'], 'penalty_point_estimate': float(penalty), 'summary': describe_cv(cv, cfg.dz_list)}
            print('independent CV', key, 'p95', CV[key]['summary']['p95'], flush=True)
        gen = torch.Generator(device=device).manual_seed(protocol['independent_J_audit_seed'] + 1)
        z = torch.randn((1, *flow.base_shape), device=device, generator=gen)
        (m, q) = flow.forward_from_base(z)
        recovered = flow.inverse_to_base(m)
        inverse_q = flow.log_prob(m)
        density = {'base_roundtrip_RMSE': float((recovered - z).square().mean().sqrt()), 'logq_forward_inverse_error_per_variable': float((q - inverse_q).abs().max() / flow.numel)}
    delta = np.array([r['paired_delta_J_per_variable'] for r in audit])
    mean = float(delta.mean())
    mcse = float(delta.std(ddof=1) / 2)
    cvdelta = CV['selected']['penalty_point_estimate'] - CV['initial']['penalty_point_estimate']
    np.savez_compressed(dest / 'full_training_responses.npz', **response_arrays, observations=problem.label.cpu().numpy(), sigma_eff=problem.sigma_eff.cpu().numpy())
    np.savez_compressed(dest / 'independent_physical_moments256.npz', **moment_arrays)
    report = {'status': 'completed', 'selected_checkpoint_sha256': sha(paths['selected']), 'initial_checkpoint_sha256': sha(paths['initial']), 'selected_training_step': payload['selected']['step'], 'changed_tensor_count': len(changed), 'changed_tensor_names': changed, 'q_identical_to_initial': not bool(changed), 'paired_independent_J_draws': 4, 'paired_J_rows': audit, 'paired_delta_J_per_variable_mean': mean, 'paired_delta_J_per_variable_MCSE': mcse, 'independent_physical_CV': CV, 'delta_CV_penalty_point_estimate': cvdelta, 'selected_full_grid_density_roundtrip': density, 'regularized_delta_point_estimate': mean + cvdelta, 'regularized_delta_uncertainty': 'Point estimate combines four-pair J integral with256-draw moment penalties; J MCSE excludes moment-penalty uncertainty; total MCSE not estimated', 'no_new_calibration_claim': True, 'full_3D_samples_no_truncation': True, 'elapsed_seconds': time.perf_counter() - started, 'protocol_sha256': sha(protocol_path), 'source_sha256': {str(p.resolve()): sha(p) for p in [Path(__file__), *paths.values()]}, 'output_sha256': {n: sha(dest / n) for n in ['full_training_responses.npz', 'independent_physical_moments256.npz']}}
    (dest / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ['status', 'changed_tensor_count', 'paired_delta_J_per_variable_mean', 'paired_delta_J_per_variable_MCSE', 'regularized_delta_point_estimate', 'elapsed_seconds']}), flush=True)
if __name__ == '__main__':
    main()
