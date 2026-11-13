from __future__ import annotations
import base64
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from cv_constraints import describe_cv

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

@torch.no_grad()
def collect_moment_cache(flow, cfg, out, device, selected_step, core_path):
    folder = out / 'posterior'
    folder.mkdir(exist_ok=True)
    here = Path(core_path).resolve().parent
    n = cfg.n_posterior_samples
    subset_n = cfg.cv_quantile_samples
    if not 2 <= subset_n <= n:
        raise ValueError('Invalid independent quantile subset')
    subset_path = folder / 'posterior_samples_quantiles256_log10rho.npy'
    cache_path = folder / 'sampling_moments_cache.json'
    sources = {name: file_hash(here / name) for name in ('AEM_Variational_Bayes_3D_FD_v8.py', 'AEMEFDO3D_fre.py', 'field_interfaces.py', 'cv_constraints.py', 'cv_export.py')}
    binding = {'checkpoint_sha256': file_hash(out / 'best.pt'), 'source_sha256': sources, 'draws': n, 'subset_draws': subset_n, 'seed': cfg.seed + 200003, 'batch_size': cfg.posterior_sample_batch, 'selected_step': selected_step, 'model_shape': list(flow.shape3)}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text('utf-8'))
        if cache['binding'] != binding:
            raise ValueError('Frozen moment cache belongs to another flow/source/sampling protocol')
        for (name, expected) in cache['moment_file_sha256'].items():
            if file_hash(folder / name) != expected:
                raise ValueError('Cached moment file changed: ' + name)
        if file_hash(subset_path) != cache['subset_npy_sha256']:
            raise ValueError('Cached independent subset changed')
        print('[Moment cache] verified; reuse the original independent1024 stream', flush=True)
        return cache
    if subset_path.exists():
        raise FileExistsError('Incomplete prior draw stream: preserve or archive before retry')
    subset = np.lib.format.open_memmap(subset_path, mode='w+', dtype='float32', shape=(subset_n, cfg.ny, cfg.nx, cfg.nz))
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 200003)
    initial_state = generator.get_state().cpu().numpy().tobytes()
    sums = {key: torch.zeros(flow.shape3, device=device, dtype=torch.float64) for key in ('rho', 'rho_square', 'm', 'm_square')}
    payload = hashlib.sha256()
    tick = time.perf_counter()
    for first in range(0, n, cfg.posterior_sample_batch):
        (m, _) = flow.sample_and_log_prob(min(cfg.posterior_sample_batch, n - first), device, antithetic=False, generator=generator)
        values = m[:, 0].double()
        rho = torch.exp(values * math.log(10.0))
        sums['rho'] += rho.sum(0)
        sums['rho_square'] += rho.square().sum(0)
        sums['m'] += values.sum(0)
        sums['m_square'] += values.square().sum(0)
        array = m[:, 0].cpu().numpy().astype('float32', copy=False)
        payload.update(array.tobytes(order='C'))
        if first < subset_n:
            subset[first:min(first + len(m), subset_n)] = array[:min(len(m), subset_n - first)]
        if first == 0 or (first + len(m)) % 128 == 0:
            print('[Independent draw stream]', first + len(m), '/', n, flush=True)
    subset.flush()
    sampling_seconds = time.perf_counter() - tick
    moment_files = {}
    for (suffix, sum_key, square_key) in [('rho', 'rho', 'rho_square'), ('log10', 'm', 'm_square')]:
        mean = sums[sum_key] / n
        variance = ((sums[square_key] - sums[sum_key].square() / n) / (n - 1)).clamp_min(0.0)
        for (label, value) in [('mean', mean), ('std', variance.sqrt())]:
            name = label + '_' + suffix + '.npy'
            np.save(folder / name, value.cpu().numpy().astype(np.float32))
            moment_files[name] = file_hash(folder / name)
    cv = np.load(folder / 'std_rho.npy') / np.load(folder / 'mean_rho.npy')
    np.save(folder / 'relative_std_rho.npy', cv)
    moment_files['relative_std_rho.npy'] = file_hash(folder / 'relative_std_rho.npy')
    subset._mmap.close()
    cache = {'schema': 'frozen_cv_v3_independent_moment_cache', 'status': 'moments_complete_quantiles_and_EFDO_deferred', 'binding': binding, 'moment_file_sha256': moment_files, 'initial_generator_state_base64': base64.b64encode(initial_state).decode('ascii'), 'all_draw_payload_sha256': payload.hexdigest(), 'subset_npy_sha256': file_hash(subset_path), 'sample_seconds': sampling_seconds, 'std_ddof': 1, 'cv_all_grid': describe_cv(cv, cfg.dz_list), 'scope': 'Independent complete1024 physical/log means/std/CV only. First256 exact models retained for later quantiles/events. No EFDO evaluation or new optimization in this phase.'}
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[Independent moments ready]', json.dumps(cache['cv_all_grid']), flush=True)
    return cache

@torch.no_grad()
def finish_cv(flow, problem, cfg, out, selected_step, core):
    folder = out / 'posterior'
    cache = collect_moment_cache(flow, cfg, out, problem.device, selected_step, core['__file__'])
    n = cfg.n_posterior_samples
    subset_n = cfg.cv_quantile_samples
    subset_path = folder / 'posterior_samples_quantiles256_log10rho.npy'
    subset = np.load(subset_path, mmap_mode='r')
    initial_state = base64.b64decode(cache['initial_generator_state_base64'])
    sampling_seconds = cache['sample_seconds']
    stats = core['posterior_statistics'](subset)
    np.save(folder / 'quantile_subset_mean_rho.npy', stats['mean_rho'])
    np.save(folder / 'quantile_subset_std_rho.npy', stats['std_rho'])
    for name in ['mean_rho', 'std_rho', 'mean_log10', 'std_log10']:
        stats[name] = np.load(folder / (name + '.npy'))
    stats['geometric_mean_rho'] = 10.0 ** stats['mean_log10']
    stats['relative_std_rho'] = stats['std_rho'] / stats['mean_rho']
    for (key, values) in stats.items():
        np.save(folder / (key + '.npy'), values)
    core['plot_stats'](stats, cfg, folder)
    core['save_structural_probabilities'](subset, cfg, folder)
    below30 = np.zeros(flow.shape3, dtype=np.uint32)
    above300 = np.zeros(flow.shape3, dtype=np.uint32)
    for first in range(0, subset_n, 16):
        rho_chunk = np.power(10.0, np.asarray(subset[first:first + 16], dtype=np.float64))
        below30 += (rho_chunk < 30.0).sum(0, dtype=np.uint32)
        above300 += (rho_chunk > 300.0).sum(0, dtype=np.uint32)
    publication_events = out / 'publication_events.npz'
    np.savez_compressed(publication_events, probability_rho_below30=(below30 / subset_n).astype(np.float32), probability_rho_above300=(above300 / subset_n).astype(np.float32), ci90_width_rho=stats['ci90_width_rho'])
    predictive = core['predictive_summary'](subset[:cfg.final_predictive_samples], problem)
    mean_log10rho = np.log10(stats['mean_rho'])
    model = torch.tensor(mean_log10rho, device=problem.device).view(1, 1, *flow.shape3)
    response = problem.forward_models(model, np.arange(problem.n_stations))[0] + problem.bias
    mean_response = response.cpu().numpy()
    observed = problem.label.cpu().numpy()
    raw_mean_rms = float(np.sqrt(np.mean((mean_response - observed) ** 2)))
    raw_predictive_rms = float(np.sqrt(np.mean((predictive['response_mean'] - observed) ** 2)))
    polarity = np.asarray(cfg.response_polarity, dtype=np.float32)[None, :]
    raw_observed = observed * polarity
    raw_mean_response = mean_response * polarity
    raw_predictive_mean = predictive['response_mean'] * polarity
    np.savez_compressed(folder / 'representative_model_responses.npz', mean_rho=mean_response, observations=observed, sigma_eff=problem.sigma_eff.cpu().numpy(), raw_signed_observations=raw_observed, raw_signed_mean_model_response=raw_mean_response, response_convention=np.asarray('mean_rho/observations use raw observation*fixed_P; raw_signed_* restore the source signed table by multiplying P exactly once'), mean_rho_sha256=file_hash(folder / 'mean_rho.npy'))
    np.savez_compressed(folder / 'predictive_responses.npz', mean=predictive['response_mean'], std=predictive['response_std'], observations=observed, sigma_eff=problem.sigma_eff.cpu().numpy(), raw_signed_observations=raw_observed, raw_signed_predictive_mean=raw_predictive_mean, response_convention=np.asarray('mean/observations use raw observation*fixed_P; raw_signed_* restore the source signed table by multiplying P exactly once'))
    cv = describe_cv(stats['relative_std_rho'], cfg.dz_list)
    metrics = core['scalar_metrics'](predictive)
    metrics.update(selected_step=selected_step, independent_model_moment_draws=n, independent_quantile_event_draws=subset_n, actual_all_station_response_draws=cfg.final_predictive_samples, response_convention='Canonical response=raw_signed*fixed_P; explicit raw_signed_* arrays restore P exactly once; RMS invariant under this unit-modulus sign transform', station_count=problem.n_stations, physical_mean_model_raw_rms_ppm=raw_mean_rms, expected_response_raw_rms_ppm=raw_predictive_rms, cv_target=cfg.cv_target, cv_weight=cfg.cv_weight, cv_summary=cv, cv_acceptance_p95=cfg.cv_acceptance_p95, meets_declared_approximate_p95=bool(cv['p95'] <= cfg.cv_acceptance_p95), std_ddof=1, model_moment_sample_seconds=sampling_seconds, posterior_target='Original all3949 signed generalized-Bayes objective plus an explicit physical-CV moment regularizer; not the unrestricted v2 posterior and not calibrated intervals', quantile_scope='Marginal quantiles, event probabilities, structural functionals and conductance quantiles use the first256 independent complete draws; physical/log means/std/CV use all1024', variance_scope='Across independent models, not across spatial cells; all full-grid random variables retained')
    (folder / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    subset_hash = file_hash(subset_path)
    checkpoint = out / 'best.pt'
    here = Path(core['__file__']).resolve().parent
    sources = {name: file_hash(here / name) for name in ('AEM_Variational_Bayes_3D_FD_v8.py', 'AEMEFDO3D_fre.py', 'field_interfaces.py', 'cv_constraints.py', 'cv_export.py')}
    recipe = {'schema': 'orkdalen_cv_v3_streamed_sampling_recipe', 'core': str(here / 'AEM_Variational_Bayes_3D_FD_v8.py'), 'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint), 'source_sha256': sources, 'seed': cfg.seed + 200003, 'initial_generator_state_base64': base64.b64encode(initial_state).decode('ascii'), 'draws': n, 'quantile_event_subset': subset_n, 'batch_size': cfg.posterior_sample_batch, 'model_shape': list(flow.shape3), 'dtype': 'float32', 'order': 'C', 'all_draw_payload_sha256': cache['all_draw_payload_sha256'], 'subset_npy_sha256': subset_hash, 'raw_all1024_saved': False, 'subset_raw_saved': not cfg.cv_remove_quantile_raw, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name() if problem.device.type == 'cuda' else 'cpu', 'cpu_threads': torch.get_num_threads(), 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'allow_cudnn_tf32': torch.backends.cudnn.allow_tf32, 'allow_matmul_tf32': torch.backends.cuda.matmul.allow_tf32}, 'interpretation': 'Recompute models through the saved full flow; no stored model/statistic has been clipped, multiplied or compressed after sampling. Hash reproducibility must be actually checked in the same environment.'}
    (folder / 'sampling_recipe_cv_v3.json').write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding='utf-8')
    event_provenance = {'result': str(out), 'mean_rho_sha256': file_hash(folder / 'mean_rho.npy'), 'selected_checkpoint_sha256': recipe['checkpoint_sha256'], 'posterior_quantile_subset_npy_sha256': subset_hash, 'model_moment_draws': n, 'posterior_draws': subset_n, 'below_threshold_ohmm': 30.0, 'above_threshold_ohmm': 300.0, 'publication_events_sha256': file_hash(publication_events), 'scope': 'Complete3D first256 independent model draws; means/std/CV separately use1024; exact30/300 differs from legacy31.6228/316.2278 exports'}
    (out / 'publication_events.provenance.json').write_text(json.dumps(event_provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    subset._mmap.close()
    if cfg.cv_remove_quantile_raw:
        os.remove(subset_path)
    print('[CV-constrained independent validation]', json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics
