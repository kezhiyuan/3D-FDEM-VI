from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from skimage.metrics import structural_similarity
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'runs'
PLAN = BASE / 'full_posterior_postprocessing_protocol_v3.json'
LEVELS = np.array([0.5, 0.8, 0.9, 0.95])
THRESHOLDS = np.array([25.0, 31.622776601683793, 40.0, 50.0, 63.09573444801933])
PREFIXES = (128, 256, 512, 1024, 2048)
CASES = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

def safe(x):
    if isinstance(x, dict):
        return {k: safe(v) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, Path):
        return str(x)
    return x

def save(path, value):
    Path(path).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def prepare():
    if PLAN.exists():
        raise FileExistsError('Preserve the registered analysis plan.')
    save(PLAN, {'status': 'frozen_before_new_posterior_analysis', 'cases': CASES, 'source_sha256': sha(__file__), 'posterior_draws': 2048, 'sample_file': 'posterior/posterior_samples_log10rho.npy', 'prefixes': PREFIXES, 'interval_levels': LEVELS, 'five_thresholds_ohmm': THRESHOLDS, 'mean': 'arithmetic physical rho mean; never exponentiated mean log10rho', 'intervals': 'Empirical physical-rho equal-tail intervals; no inflation or smoothing', 'target_mask': 'Truth cells differing from its modal log10 background by more than1e-5; evaluation only', 'SSIM': 'same3D gaussian SSIM sigma1.5/use_sample_covarianceFalse/data_range=ptp(true physical rho); valid-center local target average', 'volume_and_centroid': 'physical dx*dy*dz weights; total regions and largest6-connected body separately', 'conductance': 'integrate each complete sample over all original depth layers, then summarize; separately report C(E[rho]) and GN', 'dependence': 'Cellwise cyclic sample permutations preserve complete empirical one-cell distributions. Diagnostic control only, not a posterior/calibration.', 'dependence_seed': 20261005, 'threshold_dependence_seed': 20261028, 'dependence_domain': 'native rectangle enclosing all input stations, all depth layers; differs explicitly from truth-covered evaluation domain', 'truth_loaded_in_preparation': False, 'operator_forward_calls': 0, 'sample_modification': False})
    print('[POSTPROCESS PLAN PREPARED CPU ONLY]', sha(PLAN), flush=True)

def largest_body(mask, weights, depth):
    (labs, count) = label(mask, generate_binary_structure(3, 1))
    if not count:
        return np.array([0.0, np.nan, 0.0])
    volumes = np.bincount(labs.ravel(), weights=weights.ravel())
    volumes[0] = 0
    main = labs == np.argmax(volumes)
    v = weights[main].sum()
    return np.array([v, (weights * depth)[main].sum() / v, 1 - v / weights[mask].sum()])

def finite_quantile(a):
    a = np.asarray(a)
    out = []
    for col in a.T:
        v = col[np.isfinite(col)]
        out.append(np.quantile(v, [0.05, 0.5, 0.95]) if len(v) else [np.nan] * 3)
    return np.asarray(out).T

def physical_geometry(case, cfg, shape):
    raw = np.loadtxt(case / cfg.data_path)
    indices = np.loadtxt(case / cfg.index_path)
    rows = indices[:, 1].astype(int)
    cols = cfg.nx - 1 - indices[:, 0].astype(int)
    design = np.c_[np.ones(len(rows)), rows, cols]
    transform = np.linalg.lstsq(design, raw[:, 1:3], rcond=None)[0]
    assert np.max(abs(design @ transform - raw[:, 1:3])) < 0.1
    (sy, sx) = (np.array([cfg.ny, cfg.nx]) - shape[:2]) // 2
    (rr, cc) = np.meshgrid(sy + np.arange(shape[0] + 1) - 0.5, sx + np.arange(shape[1] + 1) - 0.5, indexing='ij')
    x = transform[0, 0] + transform[1, 0] * rr + transform[2, 0] * cc
    y = transform[0, 1] + transform[1, 1] * rr + transform[2, 1] * cc
    return (x, y, rows, cols)

def dependence(raw, cfg, rows, cols, threshold, seed, thresholds):
    n = len(raw)
    flat = raw.reshape(n, -1)
    mask = np.zeros((cfg.ny, cfg.nx, 1), bool)
    mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    weights = (mask * cfg.dx * cfg.dy * np.asarray(cfg.dz_list)).reshape(-1)
    cells = np.flatnonzero(weights)
    offset = np.random.default_rng(seed).integers(0, n, size=flat.shape[1])
    joint = np.zeros((n, len(thresholds)))
    permuted = np.zeros_like(joint)
    draws = np.arange(n)[:, None]
    maximum_mean_difference = 0.0
    for first in range(0, len(cells), 512):
        chosen = cells[first:first + 512]
        a = np.asarray(flat[:, chosen], dtype=np.float64)
        b = a[(draws + offset[chosen]) % n, np.arange(len(chosen))[None, :]]
        maximum_mean_difference = max(maximum_mean_difference, float(abs(a.astype(float).mean(0) - b.astype(float).mean(0)).max()))
        for (j, t) in enumerate(thresholds):
            joint[:, j] += (a < np.log10(t)) @ weights[chosen]
            permuted[:, j] += (b < np.log10(t)) @ weights[chosen]
    assert np.allclose(joint.mean(0), permuted.mean(0), rtol=1e-12, atol=1e-05)
    assert maximum_mean_difference < 1e-12
    center_row = int(np.median(rows))
    center_col = int(np.median(cols))
    locations = np.array([(center_row, center_col + d) for d in range(-4, 5)])
    assert (locations >= 0).all() and (locations[:, 0] < cfg.ny).all() and (locations[:, 1] < cfg.nx).all()
    ids = np.arange(flat.shape[1]).reshape(raw.shape[1:])[locations[:, 0], locations[:, 1]].ravel()
    a = np.asarray(flat[:, ids], float)
    b = np.asarray(flat[(draws + offset[ids]) % n, ids[None]], float)
    thickness = np.asarray(cfg.dz_list)
    conductance = np.sum(thickness / 10 ** a.reshape(n, 9, cfg.nz), axis=2)
    pconductance = np.sum(thickness / 10 ** b.reshape(n, 9, cfg.nz), axis=2)
    jstd = joint.std(0, ddof=1)
    pstd = permuted.std(0, ddof=1)
    return ({'thresholds': np.asarray(thresholds), 'joint_volume': joint, 'permuted_volume': permuted, 'locations_yx': locations, 'joint_conductance': conductance, 'permuted_conductance': pconductance, 'joint_correlation': np.corrcoef(conductance, rowvar=False), 'permuted_correlation': np.corrcoef(pconductance, rowvar=False)}, {'seed': seed, 'draws': n, 'thresholds_ohmm': thresholds, 'marginals_preserved_exactly_by_cyclic_bijection': True, 'max_cell_mean_log10_difference': maximum_mean_difference, 'joint_volume_mean_m3': joint.mean(0), 'permuted_volume_mean_m3': permuted.mean(0), 'joint_volume_std_m3': jstd, 'permuted_volume_std_m3': pstd, 'permuted_to_joint_std_ratio': np.divide(pstd, jstd, out=np.full_like(pstd, np.nan), where=jstd > 0), 'calibration_claim': False, 'main_interpretation_threshold_ohmm': threshold})

def analyze(key):
    plan = json.loads(PLAN.read_text('utf8'))
    assert plan['source_sha256'] == sha(__file__)
    c = CASES[key]
    execution = json.loads(c['execution'].read_text('utf8'))
    assert execution['status'] == 'completed'
    run = c['run']
    dest = BASE / (key + '_full2048_analysis')
    if dest.exists():
        raise FileExistsError('Preserve any existing post-selection analysis.')
    start = time.perf_counter()
    dest.mkdir()
    spec = importlib.util.spec_from_file_location('full_cauchy_analysis_core_' + key, c['core'])
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.ROOT = c['case']
    cfg = core.Config(**json.loads((run / 'runtime_config.json').read_text('utf8'))['config'])
    source_cfg = {'truth_path': str(c['truth'])}
    cfg.truth_path = source_cfg['truth_path']
    (truth_log, truth_meta) = core.load_truth(cfg)
    truth = 10 ** truth_log
    shape = truth.shape
    (sy, sx) = truth_meta['crop_offset_yx']
    crop = lambda a: a[sy:sy + shape[0], sx:sx + shape[1], :]
    sample_path = run / 'posterior/posterior_samples_log10rho.npy'
    raw = np.load(sample_path, mmap_mode='r')
    assert raw.shape == (2048, cfg.ny, cfg.nx, cfg.nz) and raw.dtype == np.float32
    weights = np.broadcast_to(cfg.dx * cfg.dy * np.asarray(cfg.dz_list)[None, None, :], shape)
    depth = np.cumsum(cfg.dz_list) - np.asarray(cfg.dz_list) / 2
    background = float(core.truth_background(truth_log))
    target = ~np.isclose(truth_log, background, atol=1e-05, rtol=0.0)
    valid = np.zeros(shape, bool)
    valid[5:-5, 5:-5, 5:-5] = True
    (xe, ye, rows, cols) = physical_geometry(c['case'], cfg, np.array(shape))
    xc = (xe[:-1, :-1] + xe[1:, 1:]) / 2
    regions = {'all': np.ones(shape, bool)}
    if key == 'staggered':
        regions.update(west=np.broadcast_to(xc[:, :, None] < 0, shape), east=np.broadcast_to(xc[:, :, None] >= 0, shape))
    names = list(regions)
    flat_target = target.ravel()
    flat_w = weights.ravel()
    flat_true = truth.ravel()
    model_arrays = {k: np.empty(shape, np.float32) for k in ['mean_rho', 'std_rho', 'q05_rho', 'q95_rho', 'probability_conductor']}
    prefix_means = {n: np.empty(shape, np.float32) for n in PREFIXES}
    coverage_acc = {n: np.zeros(2) for n in PREFIXES}
    interval_acc = np.zeros((len(LEVELS), 4))
    probs = np.array([0.25, 0.1, 0.05, 0.025, 0.75, 0.9, 0.95, 0.975])
    for first in range(0, truth.size, 1024):
        ids = np.arange(first, min(first + 1024, truth.size))
        (iy, ix, iz) = np.unravel_index(ids, shape)
        a = np.asarray(raw[:, iy + sy, ix + sx, iz], float)
        rho = 10 ** a
        w = flat_w[ids]
        t = flat_target[ids]
        tr = flat_true[ids]
        vals = {'mean_rho': rho.mean(0), 'std_rho': rho.std(0, ddof=1), 'probability_conductor': (a < np.log10(c['threshold'])).mean(0)}
        fullq = np.quantile(rho, probs, axis=0)
        vals.update(q05_rho=fullq[2], q95_rho=fullq[2 + len(LEVELS)])
        for (name, v) in vals.items():
            model_arrays[name].ravel()[ids] = v
        for n in PREFIXES:
            prefix_means[n].ravel()[ids] = rho[:n].mean(0)
            q = np.quantile(rho[:n], [0.05, 0.95], axis=0)
            inside = (q[0] <= tr) & (tr <= q[1])
            coverage_acc[n] += np.array([(inside * w).sum(), (inside[t] * w[t]).sum()])
        for j in range(len(LEVELS)):
            (lo, hi) = (fullq[j], fullq[j + len(LEVELS)])
            inside = (lo <= tr) & (tr <= hi)
            interval_acc[j] += np.array([(inside * w).sum(), (inside[t] * w[t]).sum(), ((hi - lo) * w).sum(), ((hi - lo)[t] * w[t]).sum()])
    for name in ['mean_rho', 'std_rho', 'q05_rho', 'q95_rho']:
        assert np.array_equal(model_arrays[name], crop(np.load(run / 'posterior' / f'{name}.npy'))), name
    mean = model_arrays['mean_rho'].astype(float)
    (ssim, local) = structural_similarity(truth, mean, data_range=float(np.ptp(truth)), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    volumes = []
    centroids = []
    bodies = []
    conductances = []
    for first in range(0, len(raw), 16):
        a = np.asarray(raw[first:first + 16, sy:sy + shape[0], sx:sx + shape[1], :], float)
        low = a < np.log10(c['threshold'])
        v = np.stack([(low * mask * weights).sum((1, 2, 3)) for mask in regions.values()], axis=1)
        numerator = np.stack([(low * mask * weights * depth).sum((1, 2, 3)) for mask in regions.values()], axis=1)
        volumes.append(v)
        centroids.append(np.divide(numerator, v, out=np.full_like(v, np.nan), where=v > 0))
        bodies.extend((largest_body(mask, weights, depth) for mask in low))
        conductances.append(np.sum(10 ** (-a) * np.asarray(cfg.dz_list), axis=3))
    volumes = np.concatenate(volumes)
    centroids = np.concatenate(centroids)
    bodies = np.asarray(bodies)
    conductances = np.concatenate(conductances)
    cq = np.quantile(conductances, [0.05, 0.5, 0.95], axis=0)
    ct = np.sum(np.asarray(cfg.dz_list) / truth, axis=2)
    cmean = np.sum(np.asarray(cfg.dz_list) / mean, axis=2)
    assert np.min(conductances.mean(0) - cmean) > -1e-06
    true_low = truth < c['threshold']
    true_volume = np.array([(true_low * mask * weights).sum() for mask in regions.values()])
    true_z = np.array([(true_low * mask * weights * depth).sum() / v if v > 0 else np.nan for (mask, v) in zip(regions.values(), true_volume)])
    np.savez_compressed(dest / 'diagnostic_arrays.npz', truth_rho=truth, **model_arrays, x_edges_m=xe, y_edges_m=ye, depth_m=depth, z_edges_m=np.r_[0, np.cumsum(cfg.dz_list)], covered90=(model_arrays['q05_rho'] <= truth) & (truth <= model_arrays['q95_rho']), conductance_q05_q50_q95=cq, conductance_truth=ct, conductance_of_mean_rho=cmean, expected_sample_conductance=conductances.mean(0), volume_truth_m3=true_volume, centroid_depth_truth_m=true_z, regions=np.array(names))
    np.savez_compressed(dest / 'sample_functionals.npz', volume_samples_m3=volumes, centroid_depth_samples_m=centroids, largest_body_samples=bodies, conductance_samples_S=conductances, regions=np.array(names))
    mc = []
    for n in PREFIXES:
        m = prefix_means[n].astype(float)
        coverage = coverage_acc[n] / [weights.sum(), weights[target].sum()]
        mc.append({'draws': n, 'ssim_physical_rho_3d': float(structural_similarity(truth, m, data_range=float(np.ptp(truth)), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)), 'coverage90_whole_and_target': coverage, 'volume_q05_q50_q95_m3': np.quantile(volumes[:n], [0.05, 0.5, 0.95], axis=0), 'centroid_q05_q50_q95_m': finite_quantile(centroids[:n]), 'median_conductance_map_RMS_change_from_final_S': float(np.sqrt(np.mean((np.median(conductances[:n], axis=0) - cq[1]) ** 2)))})
    (dep_arrays, dep_report) = dependence(raw, cfg, rows, cols, c['threshold'], 20261005, [c['threshold']])
    np.savez_compressed(dest / 'joint_dependence.npz', **dep_arrays)
    (threshold_arrays, threshold_report) = dependence(raw, cfg, rows, cols, c['threshold'], 20261028, THRESHOLDS)
    np.savez_compressed(dest / 'five_threshold_dependence.npz', **threshold_arrays)
    gn_path = c['case'] / c['GN']
    gn_cfg = core.Config(**{**core.asdict(cfg), 'truth_path': str(gn_path)})
    (gn_log, _) = core.load_truth(gn_cfg)
    gn = 10 ** core.crop_to_truth(gn_log, truth_log)
    gc = np.sum(np.asarray(cfg.dz_list) / gn, axis=2)
    np.savez_compressed(dest / 'GN_aligned.npz', rho=gn, conductance=gc)
    response = np.load(run / 'posterior/mean_model_response.npz')
    pred = response['prediction'].astype(float)
    obs = response['observations'].astype(float)
    target_columns = target.any(2)
    conductor_columns = true_low.any(2)
    legacy_columns = (truth < 99.9).any(2)
    predictive = np.load(run / 'posterior/predictive_responses.npz')
    predictive_mean = predictive['mean'].astype(float)
    assert np.array_equal(predictive['observations'], response['observations'])
    mean_low = mean < c['threshold']
    gn_low = gn < c['threshold']
    point_volume = lambda mask: np.array([(mask * r * weights).sum() for r in regions.values()])
    point_depth = lambda mask: np.array([(mask * r * weights * depth).sum() / v if v > 0 else np.nan for (r, v) in zip(regions.values(), point_volume(mask))])
    covered90 = (model_arrays['q05_rho'] <= truth) & (truth <= model_arrays['q95_rho'])
    widths = model_arrays['q95_rho'].astype(float) - model_arrays['q05_rho'].astype(float)
    profiles = []
    for (k, z) in enumerate(depth):
        row = {'depth_m': float(z)}
        for (name, mask) in [('all', np.ones(shape, bool)), ('target', target), ('background', ~target)]:
            selected = mask[:, :, k]
            row[name + '_coverage90'] = float(covered90[:, :, k][selected].mean()) if selected.any() else None
            row[name + '_mean_width90_ohmm'] = float(widths[:, :, k][selected].mean()) if selected.any() else None
        profiles.append(row)
    report = {'status': 'completed_postselection_CPU_analysis', 'case': key, 'draws': 2048, 'target_selection_uses_truth': False, 'source_run': run, 'ssim_full_valid': ssim, 'ssim_true_anomaly_valid_centers': float(local[valid & target].mean()), 'true_anomaly_volume_weighted_rmse_ohmm': float(np.sqrt(np.average((mean - truth)[target] ** 2, weights=weights[target]))), 'raw_F_arithmetic_mean_rho_RMS_ppm': float(np.sqrt(np.mean((pred - obs) ** 2))), 'raw_posterior_predictive_mean_RMS_ppm': float(np.sqrt(np.mean((predictive_mean - obs) ** 2))), 'posterior_predictive_mean_draws': 16, 'posterior_predictive_mean_QA_scope': 'RMS recomputed from archived16-draw aggregate, not a new forward or independent reconstruction of16 component responses.', 'interval_levels': LEVELS, 'physical_interval_report_columns': ['whole_volume_coverage', 'target_volume_coverage', 'whole_width_ohmm', 'target_width_ohmm'], 'physical_interval_reports': interval_acc / np.array([weights.sum(), weights[target].sum(), weights.sum(), weights[target].sum()]), 'coverage90_anomaly_volume_weighted': float(coverage_acc[2048][1] / weights[target].sum()), 'structure_threshold_ohmm': c['threshold'], 'regions': names, 'volume_q05_q50_q95_m3': np.quantile(volumes, [0.05, 0.5, 0.95], axis=0), 'centroid_q05_q50_q95_m': finite_quantile(centroids), 'volume_truth_m3': true_volume, 'centroid_truth_m': true_z, 'volume_of_arithmetic_mean_rho_m3': point_volume(mean_low), 'centroid_of_arithmetic_mean_rho_m': point_depth(mean_low), 'GN_volume_m3': point_volume(gn_low), 'GN_centroid_depth_m': point_depth(gn_low), 'GN_largest6connected_body': largest_body(gn_low, weights, depth), 'mean_rho_largest6connected_body': largest_body(mean_low, weights, depth), 'largest6connected_body_q05_q50_q95': finite_quantile(bodies), 'largest6connected_true': largest_body(true_low, weights, depth), 'conductance_coverage90_all_columns': float(((cq[0] <= ct) & (ct <= cq[2])).mean()), 'conductance_coverage90_target_columns': float(((cq[0] <= ct) & (ct <= cq[2]))[target_columns].mean()), 'conductance_mean_rho_vs_truth_RMSE_all_columns_S': float(np.sqrt(np.mean((cmean - ct) ** 2))), 'GN_conductance_vs_truth_RMSE_all_columns_S': float(np.sqrt(np.mean((gc - ct) ** 2))), 'conductance_mean_rho_vs_truth_RMSE_true_conductor_columns_S': float(np.sqrt(np.mean((cmean - ct)[conductor_columns] ** 2))), 'GN_conductance_vs_truth_RMSE_true_conductor_columns_S': float(np.sqrt(np.mean((gc - ct)[conductor_columns] ** 2))), 'conductance_mean_rho_vs_truth_RMSE_anomaly_columns_S': float(np.sqrt(np.mean((cmean - ct)[target_columns] ** 2))), 'GN_conductance_vs_truth_RMSE_anomaly_columns_S': float(np.sqrt(np.mean((gc - ct)[target_columns] ** 2))), 'conductance_mean_rho_vs_truth_RMSE_legacy_rho_below99p9_columns_S': float(np.sqrt(np.mean((cmean - ct)[legacy_columns] ** 2))), 'GN_conductance_vs_truth_RMSE_legacy_rho_below99p9_columns_S': float(np.sqrt(np.mean((gc - ct)[legacy_columns] ** 2))), 'true_conductor_and_anomaly_column_masks_equal': bool(np.array_equal(conductor_columns, target_columns)), 'legacy_below99p9_and_anomaly_column_masks_equal': bool(np.array_equal(legacy_columns, target_columns)), 'legacy_below99p9_and_anomaly_cell_masks_equal': bool(np.array_equal(truth < 99.9, target)), 'expected_sample_conductance_vs_truth_RMSE_true_conductor_columns_S': float(np.sqrt(np.mean((conductances.mean(0) - ct)[conductor_columns] ** 2))), 'median_sample_conductance_vs_truth_RMSE_true_conductor_columns_S': float(np.sqrt(np.mean((cq[1] - ct)[conductor_columns] ** 2))), 'conductor_column_definition': 'Horizontal projection of supplied truth below the declared case threshold; all-column errors reported separately.', 'anomaly_column_definition': 'Horizontal projection of all truth cells differing from modal background; legacy_low99p9 is separately retained for strict comparison with archived conductance RMSE.', 'GN_anomaly_volume_weighted_rho_RMSE_ohmm': float(np.sqrt(np.average((gn - truth)[target] ** 2, weights=weights[target]))), 'GN_ssim_physical_rho_3d': float(structural_similarity(truth, gn, data_range=float(np.ptp(truth)), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)), 'depth_profile_physical_90_intervals': profiles, 'joint_dependence': dep_report, 'five_threshold_dependence': threshold_report, 'MC_prefix_stability': mc, 'truth_metadata': truth_meta, 'truth_background_log10': background, 'scope': 'fixed-truth descriptive inclusion; not repeated-noise calibration', 'posterior_raw_kept_for_independent_QA': True, 'processing_wall_seconds': time.perf_counter() - start, 'sources': {str(p): sha(p) for p in [Path(__file__), PLAN, c['core'], c['execution'], run / 'runtime_config.json', run / 'best.pt', sample_path, run / 'posterior/mean_model_response.npz', run / 'posterior/predictive_responses.npz', Path(truth_meta['path']), gn_path]}}
    raw._mmap.close()
    save(dest / 'metrics.json', report)
    print('[FULL CPU ANALYSIS DONE]', key, ssim, report['coverage90_anomaly_volume_weighted'], flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    prepare() if args.prepare else analyze(args.case)
