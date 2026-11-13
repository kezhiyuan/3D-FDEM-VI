from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import time
import numpy as np
HERE = Path(__file__).resolve().parent
RESOURCE_KEYS = ('ny', 'nx', 'nz', 'n_coupling_layers', 'flow_hidden_channels', 'flow_dilations', 'mask_block_sizes', 'spline_bins', 'global_transport_rank', 'flow_activation_checkpointing', 'station_batch_size', 'posterior_samples_per_step', 'frequencies_per_step', 'forward_chunk_size', 'prior_mode', 'prior_phase_shifts')

def signature(cfg):
    return {k: getattr(cfg, k) if not isinstance(cfg, dict) else cfg[k] for k in RESOURCE_KEYS}

def require_passed_probe(cfg):
    pointer_path = HERE / 'stage3_probe_pass.json'
    if not pointer_path.is_file():
        raise ValueError('Run the complete two-update stage3 GPU probe before formal training')
    pointer = json.loads(pointer_path.read_text('utf-8'))
    path = Path(pointer['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != pointer['sha256']:
        raise ValueError('Stage3 probe record changed')
    record = json.loads(path.read_text('utf-8'))
    if record['status'] != 'passed' or len(record['disposable_updates']) != 2 or record['resource_signature'] != signature(cfg):
        raise ValueError('No passing complete stage3 probe for this architecture/batch configuration')
    if record['source_core_sha256'] != hashlib.sha256((HERE / 'AEM_Variational_Bayes_3D_FD_v8.py').read_bytes()).hexdigest():
        raise ValueError('Core changed after stage3 probe; rerun the resource probe')
    for row in record['disposable_updates']:
        if row['active_couplings'] != signature(cfg)['n_coupling_layers'] or row['n_samples'] != 2 or row['n_stations'] != 66 or (row['n_frequencies'] != 5):
            raise ValueError('Probe omitted part of the complete stage3 workload')
    return {'probe': str(path), 'sha256': pointer['sha256'], 'status': 'passed'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE / 'config_signed_linear_w50.json')
    p.add_argument('--gpu-granted', action='store_true')
    args = p.parse_args()
    if not args.gpu_granted:
        p.error('No GPU work without root coordinator assignment')
    import torch
    from AEM_Variational_Bayes_3D_FD_v8 import Config, Problem, MultiscaleFlow3D, configure_runtime, set_phase, streamed_gradient, prior_energy
    from validate_protocol import validate_training_inputs
    cfg = Config(**json.loads(args.config.read_text('utf-8')))
    validate_training_inputs(cfg, require_warmstart=False)
    if not cfg.flow_activation_checkpointing or cfg.posterior_samples_per_step != 2 or cfg.station_batch_size != 66 or (cfg.frequencies_per_step != 5):
        raise ValueError('The complete2x66x5 stage3 resource test is required')
    configure_runtime(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA device unavailable')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = []
    status = 'failed'
    failure = None
    try:
        problem = Problem(cfg, torch.device('cuda'))
        flow = MultiscaleFlow3D(cfg, torch.device('cuda')).train()
        set_phase(flow, 3)
        assert flow.active_coupling_layers == cfg.n_coupling_layers
        rates = [cfg.learning_rate * x for x in (cfg.location_lr_multiplier, cfg.scale_lr_multiplier, cfg.coupling_lr_multiplier)]
        optimizer = torch.optim.Adam([{'params': [flow.global_affine.bias], 'lr': rates[0]}, {'params': [flow.global_affine.raw_log_scale, flow.transport.raw_log_scale], 'lr': rates[1]}, {'params': list(flow.layers.parameters()), 'lr': rates[2]}])
        rng = np.random.default_rng(cfg.seed + 940051)
        for k in range(2):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            stations = rng.choice(problem.n_stations, 66, replace=False)
            (m, q) = flow.sample_and_log_prob(2, problem.device)
            (phi, gradient, nrms, _) = streamed_gradient(m, stations, np.arange(5), problem, 1.0, flow.numel)
            regularizer = (prior_energy(m, cfg) + q).mean() / flow.numel
            torch.autograd.backward((m, regularizer), (gradient, None))
            norm = torch.nn.utils.clip_grad_norm_([x for x in flow.parameters() if x.requires_grad], cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            rows.append({'disposable_step': k + 1, 'seconds': time.perf_counter() - tick, 'active_couplings': flow.active_coupling_layers, 'n_samples': 2, 'n_stations': 66, 'n_frequencies': 5, 'gradient_norm': float(norm), 'batch_diagnostic_nrms': nrms, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            del m, q, phi, gradient, regularizer
        status = 'passed'
    except BaseException as exc:
        status = 'OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'failed'
        failure = repr(exc)
        raise
    finally:
        report = {'schema': 'orkdalen_v2_complete_stage3_probe', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': status, 'failure': failure, 'resource_signature': signature(cfg), 'source_core_sha256': hashlib.sha256((HERE / 'AEM_Variational_Bayes_3D_FD_v8.py').read_bytes()).hexdigest(), 'disposable_updates': rows, 'total_probe_seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU_name': torch.cuda.get_device_name(), 'GPU_total_bytes': torch.cuda.get_device_properties(0).total_memory, 'official_results_or_checkpoints_saved': False, 'interpretation': 'Resource test of complete stage3, not a fitted inversion, convergence result or final runtime estimate.'}
        path = HERE / ('stage3_probe_' + str(time.time_ns()) + '.json')
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if status == 'passed':
            (HERE / 'stage3_probe_pass.json').write_text(json.dumps({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
