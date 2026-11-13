from __future__ import annotations
import math
import time
import numpy as np
import torch

def describe_cv(cv, dz=None):
    a = np.asarray(cv, dtype=np.float64)
    result = {'p00': float(a.min()), 'p50': float(np.quantile(a, 0.5)), 'p90': float(np.quantile(a, 0.9)), 'p95': float(np.quantile(a, 0.95)), 'p99': float(np.quantile(a, 0.99)), 'max': float(a.max()), 'fraction_above_0p20': float((a > 0.2).mean()), 'fraction_above_0p22': float((a > 0.22).mean()), 'fraction_above_0p25': float((a > 0.25).mean())}
    if a.ndim == 3:
        result['per_layer_p50'] = np.quantile(a, 0.5, axis=(0, 1)).tolist()
        result['per_layer_p95'] = np.quantile(a, 0.95, axis=(0, 1)).tolist()
        result['per_layer_fraction_above_0p20'] = (a > 0.2).mean(axis=(0, 1)).tolist()
        if dz is not None:
            weights = np.asarray(dz, dtype=np.float64)[None, None, :]
            result['volume_fraction_above_0p20'] = float(((a > 0.2) * weights).sum() / (weights.sum() * a.shape[0] * a.shape[1]))
    return result

@torch.no_grad()
def independent_moments(flow, device, n, batch, generator):
    sums = torch.zeros((1, 1, *flow.shape3), device=device, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    for first in range(0, n, batch):
        (m, unused_logq) = flow.sample_and_log_prob(min(batch, n - first), device, antithetic=False, generator=generator)
        del unused_logq
        rho = torch.exp(m.double() * math.log(10.0))
        sums += rho.sum(0, keepdim=True)
        squares += rho.square().sum(0, keepdim=True)
    return (sums / n, squares / n)

def cv_penalty_from_moments(mu, nu, n, target, weight):
    cv2 = (n / (n - 1) * (nu / mu.square() - 1.0)).clamp_min(0.0)
    violation = torch.relu(cv2 - target * target)
    penalty = weight * violation.square().mean()
    return (penalty, cv2, violation)

def backward_cv_penalty(flow, device, cfg, generator, period_multiplier=1.0):
    tick = time.perf_counter()
    initial = generator.get_state().clone()
    n = cfg.cv_moment_samples
    batch = cfg.cv_batch_size
    (mu, nu) = independent_moments(flow, device, n, batch, generator)
    final = generator.get_state().clone()
    (penalty, cv2, violation) = cv_penalty_from_moments(mu, nu, n, cfg.cv_target, cfg.cv_weight)
    finite = torch.isfinite(mu).all() & torch.isfinite(nu).all() & torch.isfinite(cv2).all() & torch.isfinite(violation).all()
    ready_grads = all((p.grad is not None for p in flow.parameters() if p.requires_grad))
    if type(flow).__name__ == 'MultiscaleFlow3D' and bool(finite) and (not bool(torch.count_nonzero(violation))) and ready_grads:
        if not torch.equal(generator.get_state(), final):
            raise RuntimeError('First-pass final generator changed')
        cv = cv2.sqrt().detach().cpu().numpy()[0, 0]
        return {'cv_penalty_per_variable': float(penalty), 'cv_moment_samples': n, 'cv_target': cfg.cv_target, 'cv_weight': cfg.cv_weight, 'cv_gradient_period_multiplier': period_multiplier, 'cv_seconds': time.perf_counter() - tick, 'cv_summary': describe_cv(cv, cfg.dz_list), 'exact_zero_shortcut': True}
    outside = 2.0 * cfg.cv_weight * float(period_multiplier) * violation / violation.numel()
    generator.set_state(initial)
    factor = n / (n - 1)
    for first in range(0, n, batch):
        (m, unused_logq) = flow.sample_and_log_prob(min(batch, n - first), device, antithetic=False, generator=generator)
        del unused_logq
        with torch.no_grad():
            rho = torch.exp(m.double() * math.log(10.0))
            derivative = factor * (2.0 * rho / (n * mu.square()) - 2.0 * nu / (n * mu.pow(3)))
            derivative = outside * derivative * rho * math.log(10.0)
        if not torch.isfinite(derivative).all():
            raise FloatingPointError('nonfinite physical-CV VJP')
        torch.autograd.backward(m, derivative.to(dtype=m.dtype))
        del m, rho, derivative
    if not torch.equal(generator.get_state(), final):
        raise RuntimeError('CV two-pass generator replay disagrees')
    cv = cv2.sqrt().detach().cpu().numpy()[0, 0]
    result = {'cv_penalty_per_variable': float(penalty), 'cv_moment_samples': n, 'cv_target': cfg.cv_target, 'cv_weight': cfg.cv_weight, 'cv_gradient_period_multiplier': period_multiplier, 'cv_seconds': time.perf_counter() - tick, 'cv_summary': describe_cv(cv, cfg.dz_list)}
    return result

@torch.no_grad()
def evaluate_cv(flow, device, cfg, n=None, seed=None):
    n = cfg.cv_evaluation_samples if n is None else n
    seed = cfg.cv_evaluation_seed if seed is None else seed
    generator = torch.Generator(device=device).manual_seed(seed)
    (mu, nu) = independent_moments(flow, device, n, cfg.cv_batch_size, generator)
    (penalty, cv2, _) = cv_penalty_from_moments(mu, nu, n, cfg.cv_target, cfg.cv_weight)
    cv = cv2.sqrt().cpu().numpy()[0, 0]
    return ({'cv_penalty_per_variable': float(penalty), 'cv_evaluation_samples': n, 'cv_evaluation_seed': seed, 'cv_summary': describe_cv(cv, cfg.dz_list)}, mu[0, 0].cpu().numpy())
