from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.metrics import structural_similarity
from typing import Tuple
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
ROOT = Path(__file__).resolve().parent
JGR_ROOT = Path(os.environ['AEM_PACKAGE_ROOT']).resolve()
CHANNEL_NAMES = ('880_HCP_real', '880_HCP_imag', '980_VCX_real', '980_VCX_imag', '6606_HCP_real', '6606_HCP_imag', '7001_VCX_real', '7001_VCX_imag', '34133_HCP_real', '34133_HCP_imag')

@dataclass
class Config:
    case: str = 'Orkdalen'
    data_path: str = 'data/processed_full_all/train_observations.txt'
    index_path: str = 'data/processed_full_all/train_indices_yx.txt'
    field_manifest_path: str = 'data/processed_full_all/manifest.json'
    checkpoint_path: str = '../../03_共享正演/HDF_coord_True_final_epoch_200_30m_spacing.pkl'
    result_dir: str = ''
    initial_model_path: str = ''
    warmstart_flow_checkpoint_path: str = 'result/orkdalen_full_gaussian_20260927/best.pt'
    warmstart_boundary_augmentation: bool = False
    warmstart_likelihood_change: bool = False
    warmstart_response_convention_change: bool = False
    evaluation_seed: int = 15137
    warmstart_support_change: bool = True
    initial_resistivity: float = 100.0
    map_steps: int = 0
    map_learning_rate: float = 0.05
    map_station_batch_size: int = 132
    map_eval_every: int = 20
    map_only: bool = False
    coarse_checkpoint_path: str = ''
    ny: int = 472
    nx: int = 302
    nz: int = 20
    patch_half: int = 12
    dx: float = 30.0
    dy: float = 30.0
    dz_list: tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0)
    model_min: float = -1.0
    model_max: float = 5.0
    prior_mean: float = 2.0
    horizontal_coarsening: int = 2
    posterior_family: str = 'multiscale'
    map_refine_steps: int = 0
    map_refine_learning_rate: float = 0.025
    sensitivity_probes: int = 8
    sensitivity_stations: int = 66
    sensitivity_floor_fraction: float = 0.02
    sensitivity_scale_min: float = 0.35
    sensitivity_scale_max: float = 2.5
    global_transport_rank: int = 8
    detail_prior_scale: float = 0.5
    detail_prior_df: float = 3.0
    detail_edge_multiplier: float = 0.0
    detail_edge_reference: float = 0.15
    prior_phase_shifts: tuple = ((0, 0), (0, 1), (1, 0), (1, 1))
    likelihood_mode: str = 'gaussian'
    laplace_scale: float = 1.0
    data_weight: float = 50.0
    prior_mode: str = 'phase_averaged_student'
    tv_prior_strength: float = 26100.0
    tv_axis_weights: tuple = (2.0, 1.0, 0.5)
    tv_model_scale: float = 0.49531
    horizontal_curvature_strength: float = 1740000.0
    tv_smoothing: float = 0.0
    huber_delta: float = 0.12
    lambda0: float = 0.0
    lambda_x: float = 0.2
    lambda_y: float = 0.2
    lambda_z: float = 0.12
    relative_error: float = 0.05
    noise_floor: float = 2.0
    surrogate_error_mu_path: str = ''
    surrogate_error_std_path: str = ''
    response_polarity: tuple = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    observation_transform: str = 'absolute_value'
    height_interpolation: str = 'nearest'
    outside_height_policy: str = 'inflate'
    n_coupling_layers: int = 12
    flow_activation_checkpointing: bool = True
    flow_hidden_channels: int = 32
    flow_dilations: tuple = (1, 2, 4)
    mask_block_sizes: tuple = (1, 2, 3, 5, 7, 4)
    coordinate_conditioning: bool = True
    spline_bins: int = 8
    spline_tail_bound: float = 4.0
    spline_parameter_scale: float = 0.6
    precondition_strength_y: float = 4.0
    precondition_strength_x: float = 4.0
    precondition_strength_z: float = 2.0
    precondition_exponent: float = 0.25
    init_global_log_scale: float = -3.5
    global_log_scale_limit: float = 6.0
    antithetic_samples: bool = True
    seed: int = 36
    num_steps: int = 160
    stage1_end_step: int = 0
    stage2_end_step: int = 0
    warmup_steps: int = 0
    beta_start: float = 1.0
    learning_rate: float = 0.001
    learning_rate_schedule_steps: int = 0
    location_lr_multiplier: float = 20.0
    scale_lr_multiplier: float = 5.0
    coupling_lr_multiplier: float = 0.25
    grad_clip: float = 10.0
    station_batch_size: int = 66
    frequencies_per_step: int = 5
    posterior_samples_per_step: int = 2
    forward_chunk_size: int = 8
    log_every: int = 20
    plot_every: int = 100
    plot_representative_draws: int = 32
    plot_representative_batch: int = 4
    eval_every: int = 40
    eval_samples: int = 4
    checkpoint_every: int = 40
    save_numbered_checkpoints: bool = False
    export_vtr: bool = False
    n_posterior_samples: int = 256
    posterior_sample_batch: int = 4
    final_predictive_samples: int = 16
    z_show: int = 10
    cpu_threads: int = 4
    cudnn_benchmark: bool = True
    allow_cudnn_tf32: bool = True
    truth_path: str = ''
    cv_target: float = 0.2
    cv_weight: float = 50000.0
    cv_moment_samples: int = 32
    cv_every: int = 10
    cv_batch_size: int = 2
    cv_seed: int = 750036
    cv_evaluation_samples: int = 64
    cv_evaluation_seed: int = 760036
    cv_acceptance_p95: float = 0.23
    cv_quantile_samples: int = 256
    cv_remove_quantile_raw: bool = False
    cv_finalize: bool = True

def configure_runtime(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = cfg.allow_cudnn_tf32
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

def validate_config(c):
    if c.case != 'Orkdalen':
        raise ValueError('当前完整脚本对应 Orkdalen，请使用对应算例的配置和断点')
    if not 0 <= c.stage1_end_step <= c.stage2_end_step < c.num_steps:
        raise ValueError('require 0 <= stage1_end_step <= stage2_end_step < num_steps')
    if c.learning_rate_schedule_steps and c.learning_rate_schedule_steps < c.num_steps:
        raise ValueError('learning-rate schedule must cover every training step')
    if len(c.dz_list) != c.nz or min(c.dz_list) <= 0 or min(c.dx, c.dy) <= 0:
        raise ValueError('invalid grid dimensions or spacing')
    if c.nz != 20 or c.patch_half != 12:
        raise ValueError('the supplied EFDO checkpoint requires 25 x 25 x 20 patches')
    if not c.model_min < c.prior_mean < c.model_max:
        raise ValueError('prior_mean must lie inside the bounded support')
    if not math.isfinite(c.initial_resistivity) or c.initial_resistivity <= 0:
        raise ValueError('initial_resistivity must be finite and positive, in ohm m')
    if not c.model_min < math.log10(c.initial_resistivity) < c.model_max:
        raise ValueError('uniform halfspace must lie inside the bounded support')
    if not 1 <= c.frequencies_per_step <= 5:
        raise ValueError('frequencies_per_step must be in [1,5]')
    for key in ('num_steps', 'station_batch_size', 'posterior_samples_per_step', 'forward_chunk_size', 'eval_samples', 'n_posterior_samples', 'posterior_sample_batch', 'final_predictive_samples', 'log_every', 'checkpoint_every', 'cpu_threads', 'plot_representative_draws', 'plot_representative_batch'):
        if getattr(c, key) < 1:
            raise ValueError(key + ' must be positive')
    if c.observation_transform not in ('absolute_value', 'signed'):
        raise ValueError('invalid observation_transform')
    if c.height_interpolation not in ('nearest', 'linear') or c.outside_height_policy not in ('inflate', 'reject'):
        raise ValueError('invalid height interpolation/boundary policy')
    if c.relative_error < 0 or c.noise_floor <= 0 or c.data_weight <= 0:
        raise ValueError('noise model requires nonnegative relative error and positive floor')
    if c.plot_every < 0:
        raise ValueError('plot_every must be nonnegative; use 0 to disable intermediate output')
    if min(c.lambda0, c.lambda_x, c.lambda_y, c.lambda_z) < 0:
        raise ValueError('prior precisions must be nonnegative')
    if not 0 <= c.z_show < c.nz:
        raise ValueError('invalid z_show')
    if len(c.response_polarity) != 10 or not all((p in (-1, 1) for p in c.response_polarity)):
        raise ValueError('response_polarity must contain ten fixed signs')
    if not isinstance(c.horizontal_coarsening, int) or not 1 <= c.horizontal_coarsening <= 4:
        raise ValueError('horizontal_coarsening must be an integer in [1,4]')
    if c.likelihood_mode not in ('gaussian', 'laplace') or c.laplace_scale <= 0:
        raise ValueError('invalid likelihood or Laplace scale')
    if c.likelihood_mode == 'laplace' and c.surrogate_error_std_path:
        raise ValueError('代理误差方差校准当前仅用于Gaussian似然；不能静默用于Laplace目标')
    if c.prior_mode not in ('quadratic', 'pseudo_huber', 'tv', 'multiscale_student', 'phase_averaged_student', 'phase_conditional_student', 'phase_conditional_student_normalized'):
        raise ValueError('invalid prior_mode')
    if c.posterior_family not in ('control', 'multiscale'):
        raise ValueError('invalid posterior_family')
    if (c.posterior_family == 'multiscale' or 'student' in c.prior_mode) and (c.ny % 2 or c.nx % 2):
        raise ValueError('the invertible horizontal Haar transform requires even ny and nx')
    if c.map_refine_steps < 0 or c.map_refine_learning_rate <= 0:
        raise ValueError('invalid full-grid refinement settings')
    if c.sensitivity_probes < 0 or c.sensitivity_stations < 1:
        raise ValueError('invalid sensitivity probe settings')
    if c.global_transport_rank < 0:
        raise ValueError('global_transport_rank must be nonnegative')
    if not 0 < c.sensitivity_floor_fraction <= 1 or not 0 < c.sensitivity_scale_min <= 1 <= c.sensitivity_scale_max:
        raise ValueError('invalid sensitivity scaling limits')
    if min(c.detail_prior_scale, c.detail_prior_df, c.detail_edge_reference) <= 0 or c.detail_edge_multiplier < 0:
        raise ValueError('detail prior parameters must be positive')
    if not c.prior_phase_shifts or any((len(shift) != 2 for shift in c.prior_phase_shifts)):
        raise ValueError('prior_phase_shifts must contain horizontal (y,x) pairs')
    if c.tv_prior_strength < 0 or c.horizontal_curvature_strength < 0 or c.tv_model_scale <= 0 or (c.tv_smoothing < 0):
        raise ValueError('invalid TV prior parameters')
    if len(c.tv_axis_weights) != 3 or min(c.tv_axis_weights) < 0:
        raise ValueError('invalid TV axis weights')
    if not 0 < c.cv_target < 1 or c.cv_weight <= 0 or c.cv_moment_samples < 4 or (c.cv_every < 1) or (c.cv_batch_size < 1) or (c.cv_evaluation_samples < 4):
        raise ValueError('invalid physical-CV regularizer')
    if c.map_steps < 0 or min(c.map_learning_rate, c.map_station_batch_size, c.map_eval_every) <= 0:
        raise ValueError('invalid MAP settings')

def make_block_checker_mask(shape: Tuple[int, int, int], block_size: int, invert: bool, device, phase: Tuple[int, int, int]=(0, 0, 0)):
    (ny, nx, nz) = shape
    by = max(1, int(block_size))
    bx = max(1, int(block_size))
    bz = max(1, int(round(block_size / 2)))
    (py, px, pz) = (int(phase[0]) % by, int(phase[1]) % bx, int(phase[2]) % bz)
    iy = (torch.arange(ny, device=device) + py).view(ny, 1, 1)
    ix = (torch.arange(nx, device=device) + px).view(1, nx, 1)
    iz = (torch.arange(nz, device=device) + pz).view(1, 1, nz)
    mask = ((iy // by + ix // bx + iz // bz) % 2).float()
    if invert:
        mask = 1.0 - mask
    return mask.unsqueeze(0).unsqueeze(0)

def make_normalized_coords(shape, device):
    (ny, nx, nz) = shape
    yy = torch.linspace(-1.0, 1.0, ny, device=device).view(1, 1, ny, 1, 1).expand(1, 1, ny, nx, nz)
    xx = torch.linspace(-1.0, 1.0, nx, device=device).view(1, 1, 1, nx, 1).expand(1, 1, ny, nx, nz)
    zz = torch.linspace(-1.0, 1.0, nz, device=device).view(1, 1, 1, 1, nz).expand(1, 1, ny, nx, nz)
    return torch.cat([yy, xx, zz], dim=1)

def _orthonormal_dct_matrix(n: int, device, dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=torch.float64).view(n, 1)
    j = torch.arange(n, device=device, dtype=torch.float64).view(1, n)
    C = torch.cos(math.pi / n * (j + 0.5) * k)
    C[0] *= math.sqrt(1.0 / n)
    if n > 1:
        C[1:] *= math.sqrt(2.0 / n)
    return C.to(dtype)

def _laplacian_eigs_1d(n: int, device, dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=dtype)
    return 4.0 * torch.sin(0.5 * math.pi * k / n).pow(2)

class PriorSpectralPreconditioner3D(nn.Module):

    def __init__(self, shape3, device, strength_y=4.0, strength_x=4.0, strength_z=2.0, exponent=0.25):
        super().__init__()
        (ny, nx, nz) = shape3
        self.shape3 = tuple(shape3)
        self.register_buffer('Cy', _orthonormal_dct_matrix(ny, device))
        self.register_buffer('Cx', _orthonormal_dct_matrix(nx, device))
        self.register_buffer('Cz', _orthonormal_dct_matrix(nz, device))
        ey = _laplacian_eigs_1d(ny, device)
        ex = _laplacian_eigs_1d(nx, device)
        ez = _laplacian_eigs_1d(nz, device)
        sy = (1.0 + float(strength_y) * ey).pow(-float(exponent))
        sx = (1.0 + float(strength_x) * ex).pow(-float(exponent))
        sz = (1.0 + float(strength_z) * ez).pow(-float(exponent))
        scale = sy.view(ny, 1, 1) * sx.view(1, nx, 1) * sz.view(1, 1, nz)
        self.register_buffer('spectral_scale', scale.view(1, 1, ny, nx, nz))
        self.register_buffer('logdet_const', torch.log(scale).sum().view(1))

    @staticmethod
    def _apply_axis(x, mat, axis: int, inverse: bool):
        z = x.movedim(axis, -1)
        z = torch.matmul(z, mat if inverse else mat.transpose(0, 1))
        return z.movedim(-1, axis)

    def _dct3(self, x):
        x = self._apply_axis(x, self.Cy, 2, inverse=False)
        x = self._apply_axis(x, self.Cx, 3, inverse=False)
        x = self._apply_axis(x, self.Cz, 4, inverse=False)
        return x

    def _idct3(self, x):
        x = self._apply_axis(x, self.Cz, 4, inverse=True)
        x = self._apply_axis(x, self.Cx, 3, inverse=True)
        x = self._apply_axis(x, self.Cy, 2, inverse=True)
        return x

    def forward(self, z):
        c = self._dct3(z)
        c = c * self.spectral_scale
        y = self._idct3(c)
        return (y, self.logdet_const.expand(z.size(0)))

    def inverse(self, y):
        c = self._dct3(y)
        c = c / self.spectral_scale
        z = self._idct3(c)
        return (z, (-self.logdet_const).expand(y.size(0)))

class ResidualDilatedBlock3D(nn.Module):

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        inner = max(8, channels // 2)
        g1 = max(1, min(8, channels))
        g2 = max(1, min(8, inner))
        self.norm1 = nn.GroupNorm(g1, channels)
        self.reduce = nn.Conv3d(channels, inner, 1)
        self.norm2 = nn.GroupNorm(g2, inner)
        self.spatial = nn.Conv3d(inner, inner, 3, padding=int(dilation), dilation=int(dilation), bias=False)
        self.expand = nn.Conv3d(inner, channels, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.reduce(h)
        h = self.spatial(F.silu(self.norm2(h)))
        h = self.expand(F.silu(h))
        return x + h

class SplineConditioner3D(nn.Module):

    def __init__(self, out_channels: int, hidden_channels=32, dilations=(1, 2, 4), coordinate_conditioning=True):
        super().__init__()
        self.coordinate_conditioning = bool(coordinate_conditioning)
        in_ch = 5 if self.coordinate_conditioning else 2
        self.in_proj = nn.Conv3d(in_ch, hidden_channels, 1)
        self.blocks = nn.ModuleList([ResidualDilatedBlock3D(hidden_channels, int(d)) for d in dilations])
        self.global_proj = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(hidden_channels, hidden_channels, 1), nn.SiLU(), nn.Conv3d(hidden_channels, hidden_channels, 1))
        self.out_norm = nn.GroupNorm(max(1, min(8, hidden_channels)), hidden_channels)
        self.out = nn.Conv3d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_masked, mask, coords=None):
        mb = mask.expand(x_masked.size(0), -1, -1, -1, -1)
        pieces = [x_masked, mb]
        if self.coordinate_conditioning:
            if coords is None:
                raise RuntimeError('coordinate_conditioning=True but coords is None')
            pieces.append(coords.expand(x_masked.size(0), -1, -1, -1, -1))
        h = self.in_proj(torch.cat(pieces, dim=1))
        for blk in self.blocks:
            h = blk(h)
        h = h + self.global_proj(h)
        return self.out(F.silu(self.out_norm(h)))

def _search_sorted_bins(cumvals, inputs):
    idx = torch.sum(inputs.unsqueeze(-1) >= cumvals[..., 1:], dim=-1)
    return idx.clamp(min=0, max=cumvals.shape[-1] - 2)

def _gather_last(values, idx):
    return values.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tail_bound=4.0, min_bin_width=0.001, min_bin_height=0.001, min_derivative=0.001):
    K = unnormalized_widths.shape[-1]
    if min_bin_width * K >= 1.0 or min_bin_height * K >= 1.0:
        raise ValueError('spline minimum bin size too large')
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * K) * widths
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * K) * heights
    deriv_bias = math.log(math.exp(1.0 - min_derivative) - 1.0)
    d_inner = min_derivative + F.softplus(unnormalized_derivatives + deriv_bias)
    one = torch.ones_like(d_inner[..., :1])
    derivatives = torch.cat([one, d_inner, one], dim=-1)
    left = bottom = -float(tail_bound)
    right = top = float(tail_bound)
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0)
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0)
    cumwidths = left + (right - left) * cumwidths
    cumheights = bottom + (top - bottom) * cumheights
    widths = (right - left) * widths
    heights = (top - bottom) * heights
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    inside = (inputs >= left) & (inputs <= right)
    inp = torch.where(inside, inputs, torch.zeros_like(inputs))
    (cw, ch, w, h, d) = (cumwidths, cumheights, widths, heights, derivatives)
    bin_idx = _search_sorted_bins(ch if inverse else cw, inp)
    xk = _gather_last(cw, bin_idx)
    yk = _gather_last(ch, bin_idx)
    wk = _gather_last(w, bin_idx)
    hk = _gather_last(h, bin_idx)
    dk = _gather_last(d, bin_idx)
    dk1 = _gather_last(d, bin_idx + 1)
    delta = hk / wk
    if inverse:
        y_minus = inp - yk
        a = y_minus * (dk + dk1 - 2.0 * delta) + hk * (delta - dk)
        b = hk * dk - y_minus * (dk + dk1 - 2.0 * delta)
        c = -delta * y_minus
        disc = torch.clamp(b.pow(2) - 4.0 * a * c, min=1e-12)
        root = 2.0 * c / (-b - torch.sqrt(disc))
        b_safe = torch.where(torch.abs(b) < 1e-12, torch.full_like(b, 1e-12), b)
        root_linear = -c / b_safe
        theta = torch.where(torch.abs(a) < 1e-07, root_linear, root).clamp(0.0, 1.0)
        out = xk + theta * wk
    else:
        theta = (inp - xk) / wk
        theta_om = theta * (1.0 - theta)
        numerator = hk * (delta * theta.pow(2) + dk * theta_om)
        denominator = delta + (dk + dk1 - 2.0 * delta) * theta_om
        out = yk + numerator / denominator
    theta_om = theta * (1.0 - theta)
    denominator = delta + (dk + dk1 - 2.0 * delta) * theta_om
    derivative_numerator = delta.pow(2) * (dk1 * theta.pow(2) + 2.0 * delta * theta_om + dk * (1.0 - theta).pow(2))
    lad = torch.log(torch.clamp(derivative_numerator, min=1e-12)) - 2.0 * torch.log(torch.clamp(denominator, min=1e-12))
    return (torch.where(inside, out, inputs), torch.where(inside, -lad if inverse else lad, torch.zeros_like(lad)))

class RQSCoupling3D(nn.Module):

    def __init__(self, mask, coords, hidden_channels=32, dilations=(1, 2, 4), bins=8, tail_bound=4.0, parameter_scale=0.6, coordinate_conditioning=True):
        super().__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('coords', coords)
        self.register_buffer('active_indices', (mask.flatten() < 0.5).nonzero().flatten())
        self.bins = int(bins)
        self.tail_bound = float(tail_bound)
        self.parameter_scale = float(parameter_scale)
        self.n_params = 3 * self.bins - 1
        self.net = SplineConditioner3D(self.n_params, hidden_channels=hidden_channels, dilations=dilations, coordinate_conditioning=coordinate_conditioning)

    def _params_active(self, x_masked):
        p = self.net(x_masked, self.mask, self.coords) * self.parameter_scale
        active = self.active_indices
        p = p.flatten(2).transpose(1, 2).index_select(1, active)
        K = self.bins
        uw = p[..., :K]
        uh = p[..., K:2 * K]
        ud = p[..., 2 * K:]
        return (active, uw, uh, ud)

    def forward(self, x):
        x_masked = x * self.mask
        (active, uw, uh, ud) = self._params_active(x_masked)
        xa = x.flatten(1).index_select(1, active)
        (ya, lad) = rational_quadratic_spline(xa, uw, uh, ud, inverse=False, tail_bound=self.tail_bound)
        y = x.flatten(1).index_copy(1, active, ya).view_as(x)
        logdet = lad.double().sum(dim=1)
        return (y, logdet)

    def inverse(self, y):
        y_masked = y * self.mask
        (active, uw, uh, ud) = self._params_active(y_masked)
        ya = y.flatten(1).index_select(1, active)
        (xa, lad_inv) = rational_quadratic_spline(ya, uw, uh, ud, inverse=True, tail_bound=self.tail_bound)
        x = y.flatten(1).index_copy(1, active, xa).view_as(y)
        logdet_inv = lad_inv.double().sum(dim=1)
        return (x, logdet_inv)

class SpatialAffine3D(nn.Module):

    def __init__(self, shape3, init_bias=0.0, init_log_scale=-1.2, log_scale_limit=3.0, channels=1):
        super().__init__()
        self.shape3 = tuple(shape3)
        self.log_scale_limit = float(log_scale_limit)
        self.bias = nn.Parameter(torch.full((1, channels, *self.shape3), float(init_bias)))
        x = max(min(float(init_log_scale) / self.log_scale_limit, 0.999), -0.999)
        raw0 = self.log_scale_limit * math.atanh(x)
        self.raw_log_scale = nn.Parameter(torch.full((1, channels, *self.shape3), float(raw0)))

    def effective_log_scale(self):
        return self.log_scale_limit * torch.tanh(self.raw_log_scale / self.log_scale_limit)

    def forward(self, x):
        ls = self.effective_log_scale()
        y = x * torch.exp(ls) + self.bias
        ld = ls.double().flatten().sum().expand(x.size(0))
        return (y, ld)

    def inverse(self, y):
        ls = self.effective_log_scale()
        x = (y - self.bias) * torch.exp(-ls)
        ld = (-ls).double().flatten().sum().expand(y.size(0))
        return (x, ld)

class BoundedTransform(nn.Module):

    def __init__(self, low: float, high: float, numel: int):
        super().__init__()
        if not high > low:
            raise ValueError('high must be > low')
        self.low = float(low)
        self.high = float(high)
        self.range = float(high - low)
        self.numel = int(numel)

    def forward(self, u):
        sig = torch.sigmoid(u)
        m = self.low + self.range * sig
        elem = math.log(self.range) + F.logsigmoid(u) + F.logsigmoid(-u)
        return (m, elem.double().flatten(1).sum(dim=1))

    def inverse(self, m):
        y = (m - self.low) / self.range
        y = torch.clamp(y, 1e-07, 1.0 - 1e-07)
        u = torch.log(y) - torch.log1p(-y)
        elem = math.log(self.range) + F.logsigmoid(u) + F.logsigmoid(-u)
        return (u, -elem.double().flatten(1).sum(dim=1))

def control_shape(cfg):
    factor = cfg.horizontal_coarsening
    return (math.ceil(cfg.ny / factor), math.ceil(cfg.nx / factor), cfg.nz)

def resize_model(model, shape):
    if tuple(model.shape[-3:]) == tuple(shape):
        return model
    return F.interpolate(model, size=shape, mode='trilinear', align_corners=False)

class SpatialFlow3D(nn.Module):

    def __init__(self, cfg: Config, device):
        super().__init__()
        self.cfg = cfg
        self.shape3 = (cfg.ny, cfg.nx, cfg.nz)
        self.control_shape3 = control_shape(cfg)
        self.numel = math.prod(self.control_shape3)
        self.prior_preconditioner = PriorSpectralPreconditioner3D(self.control_shape3, device=device, strength_y=cfg.precondition_strength_y, strength_x=cfg.precondition_strength_x, strength_z=cfg.precondition_strength_z, exponent=cfg.precondition_exponent)
        y = (cfg.prior_mean - cfg.model_min) / (cfg.model_max - cfg.model_min)
        y = min(max(y, 0.0001), 1.0 - 0.0001)
        init_bias = math.log(y / (1.0 - y))
        self.global_affine = SpatialAffine3D(self.control_shape3, init_bias=init_bias, init_log_scale=cfg.init_global_log_scale)
        coords = make_normalized_coords(self.control_shape3, device)
        blocks = list(cfg.mask_block_sizes)
        layers = []
        if cfg.n_coupling_layers < 2 or cfg.n_coupling_layers % 2:
            raise ValueError('n_coupling_layers must be positive and even for complementary pairs')
        for k in range(cfg.n_coupling_layers):
            pair = k // 2
            b = blocks[pair % len(blocks)]
            bz = max(1, int(round(b / 2)))
            phase = ((2 * pair + 1) % max(1, b), (3 * pair + 1) % max(1, b), (5 * pair + 1) % bz)
            mask = make_block_checker_mask(self.control_shape3, b, invert=k % 2 == 1, device=device, phase=phase)
            base_d = list(cfg.flow_dilations)
            shift = k % max(1, len(base_d))
            dil = tuple(base_d[shift:] + base_d[:shift])
            layers.append(RQSCoupling3D(mask, coords, hidden_channels=cfg.flow_hidden_channels, dilations=dil, bins=cfg.spline_bins, tail_bound=cfg.spline_tail_bound, parameter_scale=cfg.spline_parameter_scale, coordinate_conditioning=cfg.coordinate_conditioning))
        self.layers = nn.ModuleList(layers)
        self.bound = BoundedTransform(cfg.model_min, cfg.model_max, self.numel)
        self.active_coupling_layers = len(self.layers)

    def set_active_coupling_layers(self, n_active: int):
        n = int(n_active)
        if n < 0 or n > len(self.layers):
            raise ValueError(f'active coupling layers must be in [0,{len(self.layers)}], got {n}')
        self.active_coupling_layers = n

    @staticmethod
    def base_log_prob(z):
        return -0.5 * (z.double().pow(2) + math.log(2.0 * math.pi)).flatten(1).sum(dim=1)

    def forward_from_base(self, z):
        (x, ld0) = self.prior_preconditioner(z)
        total_ld = ld0.double()
        for k in range(self.active_coupling_layers):
            (x, d) = self.layers[k](x)
            total_ld = total_ld + d
        (x, d) = self.global_affine(x)
        total_ld = total_ld + d
        (m, d) = self.bound(x)
        total_ld = total_ld + d
        logq = self.base_log_prob(z) - total_ld
        return (m, logq)

    def sample_and_log_prob(self, n: int, device, antithetic=None, generator=None):
        use_pairs = self.cfg.antithetic_samples if antithetic is None else antithetic
        if use_pairs and n >= 2:
            h = n // 2
            z0 = torch.randn(h, 1, *self.control_shape3, device=device, generator=generator)
            z = torch.cat([z0, -z0], dim=0)
            if z.size(0) < n:
                z = torch.cat([z, torch.randn(1, 1, *self.control_shape3, device=device, generator=generator)], dim=0)
        else:
            z = torch.randn(n, 1, *self.control_shape3, device=device, generator=generator)
        (controls, logq) = self.forward_from_base(z)
        return (resize_model(controls, self.shape3), logq)

    def log_prob(self, m):
        if tuple(m.shape[-3:]) != self.control_shape3:
            raise ValueError('log_prob expects control-grid values, not interpolated fine-grid models')
        (x, ld_inv) = self.bound.inverse(m)
        (x, d) = self.global_affine.inverse(x)
        total_ld_inv = ld_inv + d
        for k in range(self.active_coupling_layers - 1, -1, -1):
            (x, d) = self.layers[k].inverse(x)
            total_ld_inv = total_ld_inv + d
        (z, d) = self.prior_preconditioner.inverse(x)
        total_ld_inv = total_ld_inv + d
        valid = ((m > self.bound.low) & (m < self.bound.high)).flatten(1).all(1)
        return torch.where(valid, self.base_log_prob(z) + total_ld_inv, torch.full_like(total_ld_inv, -torch.inf))

    def inverse_to_base(self, m):
        if tuple(m.shape[-3:]) != self.control_shape3:
            raise ValueError('inverse_to_base expects control-grid values')
        (x, _) = self.bound.inverse(m)
        (x, _) = self.global_affine.inverse(x)
        for k in range(self.active_coupling_layers - 1, -1, -1):
            (x, _) = self.layers[k].inverse(x)
        (z, _) = self.prior_preconditioner.inverse(x)
        return z

def haar_analysis(m):
    if m.shape[1] != 1 or m.shape[2] % 2 or m.shape[3] % 2:
        raise ValueError('Haar analysis requires one channel and even horizontal dimensions')
    (a, b, c, d) = (m[:, :, ::2, ::2], m[:, :, ::2, 1::2], m[:, :, 1::2, ::2], m[:, :, 1::2, 1::2])
    return torch.cat(((a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2), dim=1)

def haar_synthesis(bands):
    if bands.shape[1] != 4:
        raise ValueError('Haar synthesis requires all four coefficient bands')
    (a, b, c, d) = bands.split(1, dim=1)
    out = bands.new_empty((len(bands), 1, 2 * bands.shape[2], 2 * bands.shape[3], bands.shape[4]))
    out[:, :, ::2, ::2] = (a + b + c + d) / 2
    out[:, :, ::2, 1::2] = (a - b + c - d) / 2
    out[:, :, 1::2, ::2] = (a + b - c - d) / 2
    out[:, :, 1::2, 1::2] = (a - b - c + d) / 2
    return out

def sensitivity_geometry(diagonal, cfg):
    if not torch.isfinite(diagonal).all() or (diagonal < 0).any():
        raise ValueError('sensitivity diagonal must be finite and nonnegative')
    positive = diagonal[diagonal > 0]
    if positive.numel() == 0:
        return (torch.ones_like(diagonal), torch.zeros_like(diagonal))
    ref = positive.median().clamp_min(torch.finfo(diagonal.dtype).tiny)
    floor = cfg.sensitivity_floor_fraction * ref
    relative = (diagonal + floor) / (ref + floor)
    scale = relative.pow(-0.25).clamp(cfg.sensitivity_scale_min, cfg.sensitivity_scale_max)
    context = relative.log().clamp(-8.0, 8.0) / 4.0
    return (scale, context)

class BandRQSCoupling3D(nn.Module):

    def __init__(self, mask, coords, cfg):
        super().__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('coords', coords)
        self.register_buffer('sensitivity_context', torch.zeros_like(mask))
        self.register_buffer('active_indices', (mask.flatten() < 0.5).nonzero().flatten())
        self.bins = cfg.spline_bins
        self.n_params = 3 * self.bins - 1
        self.tail_bound = cfg.spline_tail_bound
        self.parameter_scale = cfg.spline_parameter_scale
        h = cfg.flow_hidden_channels
        self.in_proj = nn.Conv3d(15, h, 1)
        self.blocks = nn.Sequential(*(ResidualDilatedBlock3D(h, d) for d in cfg.flow_dilations))
        self.out = nn.Conv3d(h, 4 * self.n_params, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _parameters_active(self, x):
        b = len(x)
        context = torch.cat((x * self.mask, self.mask.expand(b, -1, -1, -1, -1), self.coords.expand(b, -1, -1, -1, -1), self.sensitivity_context.expand(b, -1, -1, -1, -1)), dim=1)
        p = self.out(self.blocks(F.silu(self.in_proj(context)))) * self.parameter_scale
        p = p.view(b, 4, self.n_params, *x.shape[2:]).permute(0, 1, 3, 4, 5, 2).reshape(b, -1, self.n_params)
        p = p.index_select(1, self.active_indices)
        return (p[..., :self.bins], p[..., self.bins:2 * self.bins], p[..., 2 * self.bins:])

    def _transform(self, x, inverse):
        (uw, uh, ud) = self._parameters_active(x)
        values = x.flatten(1).index_select(1, self.active_indices)
        (result, ld) = rational_quadratic_spline(values, uw, uh, ud, inverse=inverse, tail_bound=self.tail_bound)
        y = x.flatten(1).index_copy(1, self.active_indices, result).view_as(x)
        return (y, ld.double().sum(1))

    def forward(self, x):
        return self._transform(x, False)

    def inverse(self, x):
        return self._transform(x, True)

class SensitivityGlobalTransport(nn.Module):

    def __init__(self, numel, rank, device):
        super().__init__()
        rank = min(rank, numel)
        self.register_buffer('basis', torch.zeros(numel, rank, device=device))
        self.register_buffer('active', torch.zeros(rank, device=device))
        self.raw_log_scale = nn.Parameter(torch.zeros(rank, device=device))
        self.limit = 2.5

    def effective_log_scale(self):
        return self.limit * torch.tanh(self.raw_log_scale / self.limit) * self.active

    @torch.no_grad()
    def configure(self, sketch, diagonal_scale, initial_log_scale):
        self.basis.zero_()
        self.active.zero_()
        self.raw_log_scale.zero_()
        if not self.basis.shape[1] or sketch is None or (not len(sketch)):
            return
        matrix = sketch.flatten(1).double() * diagonal_scale.flatten().double()
        (eigenvalues, eigenvectors) = torch.linalg.eigh(matrix @ matrix.T)
        keep = (eigenvalues > max(1e-12, float(eigenvalues.max()) * 1e-10)).nonzero().flatten()
        count = min(len(keep), self.basis.shape[1])
        if not count:
            return
        indices = keep[-count:]
        values = eigenvalues[indices]
        basis = matrix.T @ eigenvectors[:, indices] / values.sqrt()[None, :]
        self.basis[:, :count].copy_(basis.to(self.basis.dtype))
        self.active[:count] = 1.0
        initial = (-0.5 * torch.log1p(values / len(sketch) * math.exp(2 * initial_log_scale))).clamp(-0.99 * self.limit, 0.0)
        self.raw_log_scale[:count].copy_(self.limit * torch.atanh(initial / self.limit))

    def _transform(self, x, inverse):
        ls = self.effective_log_scale() * (-1 if inverse else 1)
        flat = x.flatten(1)
        projected = flat @ self.basis
        y = flat + projected * torch.expm1(ls) @ self.basis.T
        return (y.reshape_as(x), ls.double().sum().expand(len(x)))

    def forward(self, x):
        return self._transform(x, False)

    def inverse(self, x):
        return self._transform(x, True)

class MultiscaleFlow3D(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.shape3 = (cfg.ny, cfg.nx, cfg.nz)
        self.band_shape = (cfg.ny // 2, cfg.nx // 2, cfg.nz)
        self.numel = math.prod(self.shape3)
        self.base_shape = (4, *self.band_shape)
        self.prior_preconditioner = PriorSpectralPreconditioner3D(self.band_shape, device, cfg.precondition_strength_y, cfg.precondition_strength_x, cfg.precondition_strength_z, cfg.precondition_exponent)
        self.global_affine = SpatialAffine3D(self.band_shape, init_log_scale=cfg.init_global_log_scale, channels=4, log_scale_limit=cfg.global_log_scale_limit)
        self.register_buffer('sensitivity_scale', torch.ones((1, *self.base_shape), device=device))
        self.register_buffer('fisher_diagonal', torch.zeros((1, *self.base_shape), device=device))
        self.transport = SensitivityGlobalTransport(self.numel, cfg.global_transport_rank, device)
        coords = make_normalized_coords(self.band_shape, device)
        dz = torch.as_tensor(cfg.dz_list, device=device)
        depth = dz.cumsum(0) - 0.5 * dz
        coords[:, 2] = ((depth - depth[0]) / (depth[-1] - depth[0]) * 2 - 1).view(1, 1, 1, -1)
        layers = []
        for k in range(cfg.n_coupling_layers):
            pair = k // 2
            if pair % 2 == 0:
                mask = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device).view(1, 4, 1, 1, 1).expand(1, 4, *self.band_shape).clone()
                mask = mask.roll(pair // 2, dims=1)
            else:
                block = cfg.mask_block_sizes[pair % len(cfg.mask_block_sizes)]
                base = make_block_checker_mask(self.band_shape, block, False, device, phase=(pair, pair + 1, 0))
                mask = torch.cat((base, 1 - base, 1 - base, base), dim=1)
            if k % 2:
                mask = 1 - mask
            layers.append(BandRQSCoupling3D(mask, coords, cfg))
        self.layers = nn.ModuleList(layers)
        self.active_coupling_layers = len(layers)
        self.bound = BoundedTransform(cfg.model_min, cfg.model_max, self.numel)
        self.to(device)
        self.initialize_from_model(torch.full((1, 1, *self.shape3), cfg.prior_mean, device=device))
    set_active_coupling_layers = SpatialFlow3D.set_active_coupling_layers
    base_log_prob = staticmethod(SpatialFlow3D.base_log_prob)

    @torch.no_grad()
    def initialize_from_model(self, model, diagonal=None, sketch=None):
        model = resize_model(model, self.shape3)
        self.global_affine.bias.copy_(haar_analysis(model_logits(model, self.cfg)))
        if diagonal is not None:
            (scale, context) = sensitivity_geometry(diagonal, self.cfg)
            self.fisher_diagonal.copy_(diagonal)
            self.sensitivity_scale.copy_(scale)
            for layer in self.layers:
                layer.sensitivity_context.copy_(context)
            self.transport.configure(sketch, scale, self.cfg.init_global_log_scale)

    def _spectral(self, x, inverse=False):
        b = len(x)
        fn = self.prior_preconditioner.inverse if inverse else self.prior_preconditioner
        (y, ld) = fn(x.reshape(b * 4, 1, *self.band_shape))
        return (y.reshape_as(x), ld.double().reshape(b, 4).sum(1))

    def forward_from_base(self, z):
        (x, ld) = self._spectral(z)
        for k in range(self.active_coupling_layers):
            if self.training and torch.is_grad_enabled() and self.cfg.flow_activation_checkpointing:
                (x, d) = activation_checkpoint(self.layers[k], x, use_reentrant=False, preserve_rng_state=False)
            else:
                (x, d) = self.layers[k](x)
            ld = ld + d
        (x, d) = self.transport(x)
        ld = ld + d
        x = x * self.sensitivity_scale
        ld = ld + self.sensitivity_scale.double().log().sum()
        (x, d) = self.global_affine(x)
        ld = ld + d
        (m, d) = self.bound(haar_synthesis(x))
        ld = ld + d
        return (m, self.base_log_prob(z) - ld)

    def sample_and_log_prob(self, n, device, antithetic=None, generator=None):
        pairs = self.cfg.antithetic_samples if antithetic is None else antithetic
        if pairs and n >= 2:
            z0 = torch.randn((n // 2, *self.base_shape), device=device, generator=generator)
            z = torch.cat((z0, -z0), dim=0)
            if n % 2:
                z = torch.cat((z, torch.randn((1, *self.base_shape), device=device, generator=generator)), dim=0)
        else:
            z = torch.randn((n, *self.base_shape), device=device, generator=generator)
        return self.forward_from_base(z)

    def _inverse(self, m):
        if tuple(m.shape[1:]) != (1, *self.shape3):
            raise ValueError('multiscale density expects the complete fine-grid model')
        (x, ld) = self.bound.inverse(m)
        (x, d) = self.global_affine.inverse(haar_analysis(x))
        ld = ld + d
        x = x / self.sensitivity_scale
        ld = ld - self.sensitivity_scale.double().log().sum()
        (x, d) = self.transport.inverse(x)
        ld = ld + d
        for k in range(self.active_coupling_layers - 1, -1, -1):
            (x, d) = self.layers[k].inverse(x)
            ld = ld + d
        (z, d) = self._spectral(x, True)
        return (z, ld + d)

    def inverse_to_base(self, m):
        return self._inverse(m)[0]

    def log_prob(self, m):
        (z, ld) = self._inverse(m)
        valid = ((m > self.bound.low) & (m < self.bound.high)).flatten(1).all(1)
        return torch.where(valid, self.base_log_prob(z) + ld, torch.full_like(ld, -torch.inf))

def _prior_penalty(x, cfg: Config):
    mode = str(getattr(cfg, 'prior_mode', 'quadratic')).lower()
    if mode == 'quadratic':
        return 0.5 * x.pow(2)
    if mode == 'pseudo_huber':
        d = max(float(cfg.huber_delta), 1e-06)
        return d * d * (torch.sqrt(1.0 + (x / d).pow(2)) - 1.0)
    raise ValueError(f'unknown prior_mode={cfg.prior_mode!r}')

def _coarse_edge_strength(coarse):
    edge2 = torch.zeros_like(coarse)
    count = torch.zeros_like(coarse)
    dy = (coarse[:, :, 1:] - coarse[:, :, :-1]).square()
    dx = (coarse[:, :, :, 1:] - coarse[:, :, :, :-1]).square()
    edge2[:, :, 1:] += dy
    edge2[:, :, :-1] += dy
    count[:, :, 1:] += 1
    count[:, :, :-1] += 1
    edge2[:, :, :, 1:] += dx
    edge2[:, :, :, :-1] += dx
    count[:, :, :, 1:] += 1
    count[:, :, :, :-1] += 1
    return torch.sqrt(edge2 / count.clamp_min(1) + 1e-12)

def _tv_energy(m, cfg):
    terms = (m[:, :, 1:] - m[:, :, :-1], m[:, :, :, 1:] - m[:, :, :, :-1], m[:, :, :, :, 1:] - m[:, :, :, :, :-1])
    energy = torch.zeros(len(m), dtype=m.dtype, device=m.device)
    for (weight, delta) in zip(cfg.tv_axis_weights, terms):
        delta = delta / cfg.tv_model_scale
        penalty = torch.sqrt(delta.square() + cfg.tv_smoothing ** 2) - cfg.tv_smoothing if cfg.tv_smoothing > 0 else delta.abs()
        energy = energy + cfg.tv_prior_strength * weight * penalty.flatten(1).mean(1)
    return energy

def _horizontal_curvature_energy(m, cfg):
    if cfg.horizontal_curvature_strength == 0:
        return torch.zeros(len(m), dtype=m.dtype, device=m.device)
    y = torch.diff(m, n=2, dim=2) / cfg.tv_model_scale
    x = torch.diff(m, n=2, dim=3) / cfg.tv_model_scale
    return cfg.horizontal_curvature_strength * (cfg.tv_axis_weights[0] * y.square().flatten(1).mean(1) + cfg.tv_axis_weights[1] * x.square().flatten(1).mean(1))

def prior_energy(m, cfg: Config):
    ref = 30.0
    dz = torch.as_tensor(cfg.dz_list, dtype=m.dtype, device=m.device)
    student_modes = ('multiscale_student', 'phase_averaged_student', 'phase_conditional_student', 'phase_conditional_student_normalized')
    if cfg.prior_mode == 'tv' or cfg.prior_mode in student_modes:
        original_model = m
        if cfg.prior_mode == 'tv':
            energy = _tv_energy(m, cfg)
        else:
            phases = ((0, 0),) if cfg.prior_mode == 'multiscale_student' else tuple(cfg.prior_phase_shifts)
            phase_energies = []
            for (shift_y, shift_x) in phases:
                shifted = torch.roll(original_model, (int(shift_y), int(shift_x)), dims=(2, 3))
                bands = haar_analysis(shifted)
                coarse = bands[:, :1] / 2.0
                if cfg.prior_mode in ('phase_conditional_student', 'phase_conditional_student_normalized'):
                    edge = _coarse_edge_strength(coarse)
                    ratio = edge / (edge + cfg.detail_edge_reference)
                    scale = cfg.detail_prior_scale * (1 + cfg.detail_edge_multiplier * ratio)
                else:
                    scale = torch.full_like(coarse, cfg.detail_prior_scale)
                standardized = bands[:, 1:] / scale
                detail = 0.5 * (cfg.detail_prior_df + 1) * torch.log1p(standardized.square() / cfg.detail_prior_df)
                if cfg.prior_mode == 'phase_conditional_student_normalized':
                    detail = detail + torch.log(scale / cfg.detail_prior_scale)
                phase_energies.append(_tv_energy(coarse, cfg) + detail.flatten(1).sum(1))
            energy = torch.stack(phase_energies).mean(0)
            energy = energy + _horizontal_curvature_energy(original_model, cfg)
        volume = (cfg.dx * cfg.dy * dz / ref ** 3).view(1, 1, 1, 1, -1)
        return energy + 0.5 * cfg.lambda0 * ((original_model - cfg.prior_mean).square() * volume).flatten(1).sum(1)
    w0 = (cfg.dx * cfg.dy * dz / ref ** 3).view(1, 1, 1, 1, -1)
    e0 = cfg.lambda0 * (_prior_penalty(m - cfg.prior_mean, cfg) * w0).flatten(1).sum(dim=1)
    wy = (cfg.dx * dz / cfg.dy / ref).view(1, 1, 1, 1, -1)
    wx = (cfg.dy * dz / cfg.dx / ref).view(1, 1, 1, 1, -1)
    dmy = m[:, :, 1:, :, :] - m[:, :, :-1, :, :]
    dmx = m[:, :, :, 1:, :] - m[:, :, :, :-1, :]
    ey = cfg.lambda_y * (_prior_penalty(dmy, cfg) * wy).flatten(1).sum(dim=1)
    ex = cfg.lambda_x * (_prior_penalty(dmx, cfg) * wx).flatten(1).sum(dim=1)
    center_spacing = 0.5 * (dz[:-1] + dz[1:])
    wz = (cfg.dx * cfg.dy / center_spacing / ref).view(1, 1, 1, 1, -1)
    dmz = m[:, :, :, :, 1:] - m[:, :, :, :, :-1]
    ez = cfg.lambda_z * (_prior_penalty(dmz, cfg) * wz).flatten(1).sum(dim=1)
    return e0 + ey + ex + ez

def load_efdo_class(case_dir):
    spec = importlib.util.spec_from_file_location('efdo_v8_definition', case_dir / 'AEMEFDO3D_fre.py')
    module = importlib.util.module_from_spec(spec)
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        spec.loader.exec_module(module)
    np.random.set_state(np_state)
    return module.EFDO
from field_interfaces import select_height_response

class Problem:

    def __init__(self, cfg, device):
        (self.cfg, self.device) = (cfg, device)
        self.case_dir = ROOT
        self.paths = {'data': self.case_dir / cfg.data_path, 'indices': self.case_dir / cfg.index_path, 'field_manifest': self.case_dir / cfg.field_manifest_path, 'checkpoint': self.case_dir / cfg.checkpoint_path}
        self.paths['split_lock'] = ROOT / 'split_lock.json'
        self.paths['source_line_mapping'] = ROOT / 'source_line_mapping.csv'
        self.paths['station_geometry'] = ROOT / 'station_geometry.npz'
        self.paths['field_interfaces'] = ROOT / 'field_interfaces.py'
        if cfg.observation_transform == 'signed':
            self.paths['conditional_sign_lock'] = ROOT / 'conditional_sign_lock.json'
        manifest = json.loads(self.paths['field_manifest'].read_text(encoding='utf-8'))
        if manifest.get('schema') not in ('orkdalen_v8_magnitude_tile_v1', 'orkdalen_v8_magnitude_full_v2', 'orkdalen_jgr_field_v1') or manifest.get('tile_shape_yx') != [cfg.ny, cfg.nx] or manifest.get('response_transform') != cfg.observation_transform or (manifest.get('index_order') != 'local_yx_direct'):
            raise ValueError('field preprocessing manifest does not match the configured model')
        raw = np.loadtxt(self.paths['data'], ndmin=2)
        idx = np.loadtxt(self.paths['indices'], ndmin=2)
        if raw.shape[1] != 14 or idx.shape != (len(raw), 2):
            raise ValueError('expected N x 14 observations and N x 2 station indices')
        if not np.isfinite(raw).all() or not np.isfinite(idx).all():
            raise ValueError('nonfinite input data')
        if not np.allclose(idx, np.round(idx), rtol=0, atol=1e-07):
            raise ValueError('station indices must be integers; no silent truncation')
        if cfg.observation_transform == 'absolute_value' and np.any(raw[:, 4:14] < 0):
            raise ValueError('processed Orkdalen responses must be nonnegative magnitudes')
        if cfg.height_interpolation == 'nearest':
            heights = (raw[:, 3] - 30) / 10
            if np.any((heights < 0) | (heights > 6)) or not np.allclose(heights, np.round(heights), rtol=0, atol=1e-05):
                raise ValueError('nearest mode requires prepared discrete 30,40,...,90 m heights')
        else:
            if cfg.outside_height_policy == 'reject' and np.any((raw[:, 3] < 30) | (raw[:, 3] > 90)):
                raise ValueError('linear height interpolation cannot extrapolate outside 30–90 m')
            if manifest.get('height_interpolation') != 'linear':
                raise ValueError('linear mode requires a manifest retaining actual flight heights')
            heights = (np.clip(raw[:, 3], 30, 90) - 30) / 10
        self.n_stations = len(raw)
        self.coordinates = raw[:, 1:3]
        self.row = idx[:, 0].astype(int)
        self.col = idx[:, 1].astype(int)
        if min(self.row.min(), self.col.min()) < 0 or self.row.max() >= cfg.ny or self.col.max() >= cfg.nx:
            raise ValueError('station outside model')
        self.height = torch.tensor(heights if cfg.height_interpolation == 'linear' else np.round(heights), device=device, dtype=torch.float32 if cfg.height_interpolation == 'linear' else torch.long)
        label = raw[:, 4:14] * np.asarray(cfg.response_polarity)
        self.label = torch.tensor(label, device=device, dtype=torch.float32)
        self.sigma = cfg.relative_error * self.label.abs() + cfg.noise_floor
        self.bias = self._load_error(cfg.surrogate_error_mu_path, 'mean')
        self.extra_std = self._load_error(cfg.surrogate_error_std_path, 'std')
        if manifest['schema'] in ('orkdalen_v8_magnitude_full_v2', 'orkdalen_jgr_field_v1'):
            excess = np.asarray(manifest['height_excess_m_by_source_row_id'], dtype=np.float32)
            ids = raw[:, 0].astype(int)
            if len(excess) != manifest['source_station_count'] or not np.isfinite(excess).all() or np.any(excess < 0) or (not np.allclose(raw[:, 0], ids, rtol=0, atol=1e-07)) or np.any(ids < 0) or np.any(ids >= len(excess)) or (manifest.get('height_excess_error_scale_m') != 5.0):
                raise ValueError('invalid full-survey height-error metadata')
            self.height_excess_m = excess[ids]
            height_extra = self.sigma * torch.tensor(self.height_excess_m[:, None] / 5.0, device=device, dtype=torch.float32)
        else:
            self.height_excess_m = np.zeros(self.n_stations, dtype=np.float32)
            height_extra = torch.zeros_like(self.sigma)
        self.sigma_eff = (self.sigma.square() + self.extra_std.square() + height_extra.square()).sqrt()
        self.freq = torch.tensor([[2.9445], [2.9912], [3.8199], [3.8452], [4.5332]], device=device)
        cls = load_efdo_class(self.case_dir)
        self.model = cls(14, 14, 14, 36, 2, [12500, 256, 128, 64, 7], 7, 'xavier_uniform', 3, 3, 'gelu').to(device)
        self.model.load_state_dict(torch.load(self.paths['checkpoint'], map_location=device, weights_only=True))
        self.model.eval().requires_grad_(False)
        self.mean_out = torch.tensor([69.6941, 52.7563], device=device)
        self.std_out = torch.tensor([97.8719, 71.6932], device=device) + 1e-05
        off = np.arange(-cfg.patch_half, cfg.patch_half + 1)
        yy = np.clip(self.row[:, None, None] + off[None, :, None], 0, cfg.ny - 1)
        xx = np.clip(self.col[:, None, None] + off[None, None, :], 0, cfg.nx - 1)
        self.patch_indices = torch.tensor((yy * cfg.nx + xx).reshape(len(raw), -1), device=device, dtype=torch.long)

    def _load_error(self, path, name):
        if not path:
            return torch.zeros_like(self.label)
        source = (ROOT / path).resolve()
        a = np.load(source)
        if a.shape not in [(10,), (self.n_stations, 10)] or not np.isfinite(a).all():
            raise ValueError('invalid surrogate error ' + name)
        if name == 'std' and np.any(a < 0):
            raise ValueError('negative surrogate standard deviation')
        self.paths['surrogate_error_' + name] = source
        return torch.tensor(np.broadcast_to(a, self.label.shape).copy(), device=self.device, dtype=torch.float32)

    def patches(self, m, stations):
        idx = torch.as_tensor(stations, device=self.device, dtype=torch.long)
        s = m.shape[0]
        b = idx.numel()
        c = self.cfg
        gather = self.patch_indices.index_select(0, idx).flatten()
        patches = m[:, 0].reshape(s, c.ny * c.nx, c.nz).index_select(1, gather)
        patches = patches.reshape(s * b, 25, 25, 20, 1)
        return ((patches - 1.4987) / (0.4953 + 1e-05), self.height.index_select(0, idx).repeat(s))

    def forward_patches(self, patches, heights, frequencies):
        model = self.model
        fi = torch.as_tensor(frequencies, device=self.device, dtype=torch.long)
        (b, nf) = (patches.shape[0], fi.numel())
        x = model.ufno(model.BranchTrunk(patches, self.freq.index_select(0, fi)))
        p = model.padding
        x = x[:, :-p, :-p, :-p, :].contiguous()
        x = model.fc2(model.activation(model.fc1(x)))
        x = x.reshape(b, nf, -1, model.nout).permute(0, 1, 3, 2)
        for layer in model.linears[:-1]:
            x = model.activation(layer(x))
        x = model.linears[-1](x).permute(0, 1, 3, 2)
        x = x * self.std_out + self.mean_out
        x = select_height_response(x, heights, self.cfg.height_interpolation)
        return x.reshape(b, nf * 2)

    def forward_models(self, m, stations, frequencies=tuple(range(5)), chunk_size=None):
        if m.shape[1:] != (1, self.cfg.ny, self.cfg.nx, self.cfg.nz):
            raise ValueError('expected (S,1,Ny,Nx,Nz)')
        (patches, h) = self.patches(m, stations)
        chunk_size = chunk_size or self.cfg.forward_chunk_size
        chunks = [self.forward_patches(patches[k:k + chunk_size], h[k:k + chunk_size], frequencies) for k in range(0, len(patches), chunk_size)]
        return torch.cat(chunks).reshape(len(m), len(stations), 2 * len(frequencies))

    def data(self, stations, frequencies):
        si = torch.as_tensor(stations, device=self.device, dtype=torch.long)
        ci = torch.as_tensor(np.asarray(frequencies)[:, None] * 2 + np.arange(2), device=self.device, dtype=torch.long).flatten()
        return tuple((a.index_select(0, si).index_select(1, ci) for a in (self.label, self.bias, self.sigma_eff)))

def likelihood_elements(residual, sigma, cfg):
    if cfg.likelihood_mode == 'gaussian':
        return 0.5 * (residual / sigma).square()
    return residual.abs() / (sigma * cfg.laplace_scale)

def likelihood(pred, obs, bias, sigma, n_total, n_freq=5, cfg=None):
    scale = n_total / pred.shape[1] * 5 / n_freq
    energy = 0.5 * ((pred + bias - obs) / sigma).square() if cfg is None else likelihood_elements(pred + bias - obs, sigma, cfg)
    return energy.flatten(1).sum(1) * scale * (1.0 if cfg is None else cfg.data_weight)

def streamed_gradient(m, stations, frequencies, problem, beta, objective_scale):
    leaf = m.detach().requires_grad_(True)
    phi = torch.zeros(len(m), device=m.device)
    squared_mean = torch.zeros((), device=m.device, dtype=torch.float64)
    squared_samples = torch.zeros((), device=m.device, dtype=torch.float64)
    s = len(m)
    b = len(stations)
    nf = len(frequencies)
    micro = max(1, problem.cfg.forward_chunk_size // s)
    for start in range(0, b, micro):
        sub = stations[start:start + micro]
        pred = problem.forward_models(leaf, sub, frequencies)
        (obs, bias, sigma) = problem.data(sub, frequencies)
        r = (pred + bias - obs) / sigma
        part = likelihood_elements(pred + bias - obs, sigma, problem.cfg).flatten(1).sum(1) * (problem.n_stations / b) * (5 / nf) * problem.cfg.data_weight
        (beta * part.mean() / objective_scale).backward()
        phi += part.detach()
        squared_mean += ((pred.detach().mean(0) + bias - obs) / sigma).square().sum().double()
        squared_samples += r.detach().square().sum().double()
        del pred, r, part
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise FloatingPointError('missing or nonfinite EFDO model gradient')
    return (phi, leaf.grad.detach(), math.sqrt(float(squared_mean) / (b * nf * 2)), math.sqrt(float(squared_samples) / (s * b * nf * 2)))

@torch.no_grad()
def predictive_summary(samples, problem):
    stations = np.arange(problem.n_stations)
    predictions = []
    for sample in samples:
        m = torch.as_tensor(np.array(sample), device=problem.device).view(1, 1, *sample.shape)
        predictions.append(problem.forward_models(m, stations)[0].cpu().numpy())
    a = np.stack(predictions).astype(np.float64)
    obs = problem.label.cpu().numpy()
    bias = problem.bias.cpu().numpy()
    sigma = problem.sigma_eff.cpu().numpy()
    residual = (a + bias - obs) / sigma
    rms = np.sqrt(np.mean(residual ** 2, axis=(1, 2)))
    mean = a.mean(0) + bias
    energy = 0.5 * residual ** 2 if problem.cfg.likelihood_mode == 'gaussian' else np.abs(residual) / problem.cfg.laplace_scale
    return {'n_draws': len(a), 'expected_sample_rms': float(np.sqrt(np.mean(residual ** 2))), 'expected_data_energy': float(energy.sum(axis=(1, 2)).mean()), 'predictive_mean_rms': float(np.sqrt(np.mean(((mean - obs) / sigma) ** 2))), 'sample_rms': rms.tolist(), 'channel_rms': dict(zip(CHANNEL_NAMES, np.sqrt(np.mean(((mean - obs) / sigma) ** 2, axis=0)).tolist())), 'response_mean': mean.astype(np.float32), 'response_std': a.std(0, ddof=1 if len(a) > 1 else 0).astype(np.float32)}

@torch.no_grad()
def evaluate(flow, problem, cfg):
    generator = torch.Generator(device=problem.device).manual_seed(cfg.evaluation_seed)
    models = []
    logqs = []
    priors = []
    for _ in range(cfg.eval_samples):
        (m, q) = flow.sample_and_log_prob(1, problem.device, antithetic=False, generator=generator)
        models.append(m[0, 0].cpu().numpy())
        logqs.append(float(q[0]))
        priors.append(float(prior_energy(m, cfg)[0]))
    samples = np.stack(models)
    result = predictive_summary(samples, problem)
    result['negative_elbo'] = cfg.data_weight * result['expected_data_energy'] + float(np.mean(logqs)) + float(np.mean(priors))
    return (result, samples.mean(0))

def scalar_metrics(result):
    return {k: v for (k, v) in result.items() if not isinstance(v, np.ndarray)}

def estimate_sensitivity(model, problem, cfg, out):
    coeff = haar_analysis(model_logits(model.detach(), cfg)).requires_grad_(True)
    diagonal = torch.zeros_like(coeff)
    start = time.perf_counter()
    if cfg.sensitivity_probes:
        rng = np.random.default_rng(cfg.seed + 7919)
        stations = np.sort(rng.choice(problem.n_stations, min(cfg.sensitivity_stations, problem.n_stations), replace=False))
        generator = torch.Generator(device=problem.device).manual_seed(cfg.seed + 7927)
        projected = torch.zeros((cfg.sensitivity_probes, *coeff.shape[1:]), device=problem.device)
        for first in range(0, len(stations), cfg.forward_chunk_size):
            sub = stations[first:first + cfg.forward_chunk_size]
            logits = haar_synthesis(coeff)
            physical = cfg.model_min + (cfg.model_max - cfg.model_min) * torch.sigmoid(logits)
            pred = problem.forward_models(physical, sub)
            sigma = problem.data(sub, np.arange(5))[2]
            scale = sigma if cfg.likelihood_mode == 'gaussian' else cfg.laplace_scale
            for k in range(cfg.sensitivity_probes):
                signs = torch.randint(0, 2, pred.shape, device=problem.device, generator=generator).to(pred.dtype) * 2 - 1
                value = (pred * signs / scale).sum()
                gradient = torch.autograd.grad(value, coeff, retain_graph=k + 1 < cfg.sensitivity_probes)[0]
                projected[k].add_(gradient[0].detach())
            del pred, physical, logits, gradient, value
        diagonal = projected.square().mean(0, keepdim=True) * (problem.n_stations / len(stations))
    else:
        stations = np.empty(0, dtype=int)
        projected = torch.empty((0, *coeff.shape[1:]), device=problem.device)
    if not torch.isfinite(diagonal).all():
        raise FloatingPointError('nonfinite sensitivity geometry')
    (scale, context) = sensitivity_geometry(diagonal, cfg)
    anchor_sha256 = hashlib.sha256(model.detach().cpu().numpy().tobytes()).hexdigest()
    np.savez_compressed(out / 'sensitivity.npz', fisher_diagonal=diagonal.cpu().numpy(), scale=scale.cpu().numpy(), context=context.cpu().numpy(), stations=stations, anchor_sha256=np.asarray(anchor_sha256), projected_vjps=(projected * math.sqrt(problem.n_stations / max(1, len(stations)))).cpu().numpy())
    report = {'probes': cfg.sensitivity_probes, 'stations': stations.tolist(), 'likelihood': cfg.likelihood_mode, 'coordinates': 'orthonormal Haar coefficients of bounded-model logits', 'seconds': time.perf_counter() - start, 'scale_min': float(scale.min()), 'scale_max': float(scale.max()), 'anchor_model_sha256': anchor_sha256, 'scope': 'Gaussian inverse-variance Jacobian diagonal' if cfg.likelihood_mode == 'gaussian' else 'response-Jacobian diagonal without heteroscedastic weighting; not Fisher information', 'not_posterior_precision_or_covariance': True}
    (out / 'sensitivity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[Sensitivity]', report['seconds'], 'seconds; scale', report['scale_min'], report['scale_max'], flush=True)
    return diagonal.detach()

def fit_refinement(initial, diagonal, problem, cfg, out, hashes, resume_payload=None):
    folder = out / 'map_refined'
    folder.mkdir(exist_ok=True)
    (scale, _) = sensitivity_geometry(diagonal, cfg)
    coordinates = nn.Parameter(haar_analysis(model_logits(initial, cfg)) / scale)
    optimizer = torch.optim.Adam([coordinates], lr=cfg.map_refine_learning_rate)
    rng = np.random.default_rng(cfg.seed + 510019)

    def reconstruct():
        return cfg.model_min + (cfg.model_max - cfg.model_min) * torch.sigmoid(haar_synthesis(coordinates * scale))
    history = []
    first = 1
    with torch.no_grad():
        best_metrics = evaluate_point(reconstruct(), problem, cfg)
        best_coordinates = coordinates.detach().clone()
        best_step = 0
    if resume_payload is not None:
        with torch.no_grad():
            coordinates.copy_(resume_payload['coordinates'])
        optimizer.load_state_dict(resume_payload['optimizer'])
        rng.bit_generator.state = resume_payload['rng_numpy']
        best_coordinates = resume_payload['best_coordinates'].to(problem.device)
        best_metrics = resume_payload['best_metrics']
        best_step = resume_payload['best_step']
        first = resume_payload['step'] + 1
        history = resume_payload['history']
    print('[Refine initial]', best_metrics['predictive_mean_rms'], best_metrics['point_objective'], flush=True)
    for step in range(first, cfg.map_refine_steps + 1):
        tick = time.perf_counter()
        si = rng.choice(problem.n_stations, min(cfg.map_station_batch_size, problem.n_stations), replace=False)
        optimizer.param_groups[0]['lr'] = cfg.map_refine_learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, cfg.map_refine_steps))))
        optimizer.zero_grad(set_to_none=True)
        model = reconstruct()
        (phi, gradient, rms, _) = streamed_gradient(model, si, np.arange(5), problem, 1.0, problem.n_stations * 10)
        prior = prior_energy(model, cfg).mean() / (problem.n_stations * 10)
        torch.autograd.backward((model, prior), (gradient, None))
        torch.nn.utils.clip_grad_norm_([coordinates], cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'batch_rms': rms, 'full_objective': float('nan'), 'full_rms': float('nan'), 'seconds': time.perf_counter() - tick}
        del model, phi, gradient, prior
        if step % cfg.map_eval_every == 0 or step == cfg.map_refine_steps:
            with torch.no_grad():
                model = reconstruct()
                ev = evaluate_point(model, problem, cfg)
                row.update(full_objective=ev['point_objective'], full_rms=ev['predictive_mean_rms'])
                if ev['point_objective'] < best_metrics['point_objective']:
                    best_metrics = ev
                    best_coordinates = coordinates.detach().clone()
                    best_step = step
                np.save(folder / f'model_log10_step_{step:06d}.npy', model[0, 0].cpu().numpy())
            print('[Refine full]', step, ev['predictive_mean_rms'], ev['point_objective'], flush=True)
        history.append(row)
        if step % cfg.map_eval_every == 0 or step == cfg.map_refine_steps:
            atomic_save({'format': 'vi_v8_refine', 'config': asdict(cfg), 'inputs': hashes, 'step': step, 'initial': initial.detach(), 'diagonal': diagonal, 'coordinates': coordinates.detach(), 'optimizer': optimizer.state_dict(), 'rng_numpy': rng.bit_generator.state, 'best_coordinates': best_coordinates, 'best_metrics': best_metrics, 'best_step': best_step, 'history': history}, out / 'refine_last.pt')
            with (folder / 'history.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
        if step == first or step % 10 == 0:
            print('[Refine]', step, 'batch RMS', rms, flush=True)
    with torch.no_grad():
        coordinates.copy_(best_coordinates)
        model = reconstruct().detach()
        a = model[0, 0].cpu().numpy()
        np.save(folder / 'model_log10.npy', a)
        np.save(folder / 'model_rho.npy', 10.0 ** a)
        best_metrics.update(selected_step=best_step, selection='full data energy plus the declared prior; no truth', initialization='coarse MAP of the same run', training_step_seconds=sum((r['seconds'] for r in history)))
        if cfg.truth_path:
            (truth, _) = load_truth(cfg)
            best_metrics['reconstruction'] = reconstruction_metrics(a, truth, cfg)
            plot_truth_comparison(a, truth, cfg, folder / 'truth_comparison.png', cfg.z_show, 'Full-grid MAP')
        (folder / 'metrics.json').write_text(json.dumps(best_metrics, indent=2), encoding='utf-8')
        write_vtr(10.0 ** a, folder / 'model_rho.vtr', cfg, 'map_resistivity')
    return model

def atomic_save(payload, path):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)

def fingerprint(problem):
    hashes = {}
    for (name, path) in problem.paths.items():
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda : stream.read(1024 * 1024), b''):
                h.update(block)
        hashes[name] = {'path': str(path), 'sha256': h.hexdigest()}
    return hashes

def phase(step, c):
    return 1 if step <= c.stage1_end_step else 2 if step <= c.stage2_end_step else 3

def set_phase(flow, stage):
    flow.requires_grad_(False)
    flow.global_affine.bias.requires_grad_(True)
    flow.global_affine.raw_log_scale.requires_grad_(stage >= 2)
    if hasattr(flow, 'transport'):
        flow.transport.raw_log_scale.requires_grad_(stage >= 2)
    if stage == 3:
        flow.layers.requires_grad_(True)
    flow.set_active_coupling_layers(0 if stage < 3 else len(flow.layers))

def truth_background(truth):
    (values, counts) = np.unique(truth, return_counts=True)
    return float(values[np.argmax(counts)])

def load_truth(cfg):
    path = (ROOT / cfg.truth_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到真值文件：{path}')
    metadata = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'alignment': 'native row/column order; centered crop; no flips'}
    if path.suffix.lower() == '.rho':
        lines = [s.split('#')[0].strip() for s in path.read_text(encoding='utf-8-sig').splitlines()]
        lines = [s for s in lines if s]
        header = lines[0].split()
        if len(header) < 5 or header[-1].upper() != 'LOGE':
            raise ValueError('当前 .rho 读取器要求明确的 LOGE（自然对数电阻率）头')
        (nx, ny, nz) = map(int, header[:3])
        if min(nx, ny, nz) <= 0:
            raise ValueError('.rho 网格数必须为正')
        values = np.array([float(t.replace('D', 'E').replace('d', 'e')) for t in ' '.join(lines[1:]).split()], dtype=np.float64)
        offset = nx + ny + nz
        count = nx * ny * nz
        if len(values) not in (offset + count, offset + count + 4):
            raise ValueError('.rho 网格数、模型值数量或末尾原点/旋转字段不符')
        (dx, dy, dz) = (values[:nx], values[nx:nx + ny], values[nx + ny:offset])
        if nz != cfg.nz or not np.allclose(dx, cfg.dx) or (not np.allclose(dy, cfg.dy)) or (not np.allclose(dz, cfg.dz_list)):
            raise ValueError('真值网格间距与反演不同，需要明确重采样；不自动拉伸真值')
        truth = values[offset:offset + count].reshape(nz, ny, nx).transpose(1, 2, 0) / np.log(10.0)
        metadata['encoding'] = 'natural log resistivity converted to log10'
        metadata['file_origin_rotation'] = values[offset + count:].tolist()
    elif path.suffix.lower() == '.npy':
        truth = np.asarray(np.load(path), dtype=np.float64)
        metadata['encoding'] = 'log10 resistivity'
    else:
        raise ValueError('truth_path 只支持明确编码的 .rho 或 .npy')
    if truth.ndim != 3 or truth.shape[2] != cfg.nz or (not np.isfinite(truth).all()):
        raise ValueError('真值必须是有限的三维电阻率模型')
    delta = np.array([cfg.ny, cfg.nx]) - np.array(truth.shape[:2])
    if np.any(delta < 0) or np.any(delta % 2):
        raise ValueError('真值与反演网格不能对称中心对齐；请明确网格映射')
    (sy, sx) = (delta // 2).tolist()
    anomaly = np.abs(truth - truth_background(truth)) > 0.5
    edges = np.r_[0.0, np.cumsum(cfg.dz_list)]
    layers = np.flatnonzero(anomaly.any(axis=(0, 1)))
    metadata.update(shape_yxz=list(truth.shape), crop_offset_yx=[sy, sx], background_log10=truth_background(truth), anomaly_threshold_log10=0.5, anomaly_cells=int(anomaly.sum()), anomaly_layers_zero_based=layers.tolist(), anomaly_depth_intervals_m=[[float(edges[k]), float(edges[k + 1])] for k in layers], resistivity_range_ohm_m=[float(10 ** truth.min()), float(10 ** truth.max())])
    return (truth, metadata)

def crop_to_truth(model, truth):
    model = np.asarray(model)
    if model.ndim != 3 or model.shape[2] != truth.shape[2] or (not np.isfinite(model).all()):
        raise ValueError('待评价模型形状或数值无效')
    delta = np.array(model.shape[:2]) - np.array(truth.shape[:2])
    if np.any(delta < 0) or np.any(delta % 2):
        raise ValueError('待评价模型与真值不能中心对齐')
    (sy, sx) = (delta // 2).tolist()
    return model[sy:sy + truth.shape[0], sx:sx + truth.shape[1], :]

def reconstruction_metrics(model_log10, truth, cfg):
    model = crop_to_truth(model_log10, truth)
    residual = model - truth
    anomaly = np.abs(truth - truth_background(truth)) > 0.5
    predicted = np.abs(model - truth_background(truth)) > 0.5
    weights = np.broadcast_to(np.asarray(cfg.dz_list)[None, None, :], truth.shape)
    metrics = {'cells': int(truth.size), 'anomaly_cells': int(anomaly.sum()), 'rmse_log10_all': float(np.sqrt(np.mean(residual ** 2))), 'volume_weighted_rmse_log10': float(np.sqrt(np.sum(weights * residual ** 2) / weights.sum())), 'rmse_log10_by_layer': np.sqrt(np.mean(residual ** 2, axis=(0, 1))).tolist()}
    ssim_truth = np.asarray(truth, dtype=np.float64)
    ssim_model = np.asarray(model, dtype=np.float64)
    dynamic_range = float(np.ptp(ssim_truth))
    if dynamic_range > 0:
        metrics['ssim_log10_volume_3d'] = float(structural_similarity(ssim_truth, ssim_model, data_range=dynamic_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        ssim_by_layer = [float(structural_similarity(ssim_truth[:, :, z], ssim_model[:, :, z], data_range=dynamic_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)) for z in range(truth.shape[2])]
        anomaly_layers = np.any(anomaly, axis=(0, 1))
        metrics['ssim_log10_by_layer'] = ssim_by_layer
        metrics['ssim_log10_all_layers'] = float(np.mean(ssim_by_layer))
        metrics['ssim_log10_anomaly_layers'] = float(np.mean(np.asarray(ssim_by_layer)[anomaly_layers])) if anomaly_layers.any() else None
        metrics['ssim_log10_data_range'] = dynamic_range
    for (name, mask) in [('anomaly', anomaly), ('background', ~anomaly)]:
        metrics['rmse_log10_' + name] = float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else None
    metrics['correlation'] = float(np.corrcoef(model.ravel(), truth.ravel())[0, 1]) if model.std() > 1e-12 and truth.std() > 1e-12 else None
    for (name, target, estimate) in [('anomaly', anomaly, predicted), ('conductor', truth < truth_background(truth) - 0.5, model < truth_background(truth) - 0.5), ('resistor', truth > truth_background(truth) + 0.5, model > truth_background(truth) + 0.5)]:
        union = int(np.count_nonzero(target | estimate))
        metrics[name + '_iou'] = float(np.count_nonzero(target & estimate) / union) if union else None
    true_positive = (anomaly & predicted).sum(axis=(0, 1))
    target_count = anomaly.sum(axis=(0, 1))
    metrics['anomaly_cells_by_layer'] = target_count.tolist()
    metrics['predicted_anomaly_cells_by_layer'] = predicted.sum(axis=(0, 1)).tolist()
    metrics['anomaly_recall_by_layer'] = [float(tp / n) if n else None for (tp, n) in zip(true_positive, target_count)]
    return metrics

def spatial_coverage(lower_log10, upper_log10, truth, cfg):
    lower = crop_to_truth(lower_log10, truth)
    upper = crop_to_truth(upper_log10, truth)
    if np.any(lower > upper):
        raise ValueError('可信区间下界不能大于上界')
    inside = (truth >= lower) & (truth <= upper)
    anomaly = ~np.isclose(truth, truth_background(truth), atol=1e-05, rtol=0.0)
    masks = [('all', np.ones_like(anomaly)), ('anomaly', anomaly), ('background', ~anomaly)]
    weights = np.broadcast_to(np.asarray(cfg.dz_list)[None, None, :] * cfg.dx * cfg.dy, truth.shape)
    report = {name: float(np.average(inside[mask], weights=weights[mask])) if mask.any() else None for (name, mask) in masks}
    report['cell_coverage'] = {name: float(inside[mask].mean()) if mask.any() else None for (name, mask) in masks}
    report['coverage_weighting'] = 'physical cell volume'
    report['target_definition'] = 'all cells differing from the truth background'
    width = upper - lower
    report['mean_width_log10'] = {name: float(np.average(width[mask], weights=weights[mask])) if mask.any() else None for (name, mask) in masks}
    report['median_width_log10'] = {name: float(np.median(width[mask])) if mask.any() else None for (name, mask) in masks}
    edges = np.r_[0, np.cumsum(cfg.dz_list)]
    report['by_layer'] = []
    for z in range(truth.shape[2]):
        row = {'index': z, 'top_m': float(edges[z]), 'bottom_m': float(edges[z + 1])}
        for (name, mask) in masks:
            sub = mask[:, :, z]
            row[name] = float(inside[:, :, z][sub].mean()) if sub.any() else None
            row[name + '_mean_width_log10'] = float(width[:, :, z][sub].mean()) if sub.any() else None
        report['by_layer'].append(row)
    return report

def plot_truth_comparison(model_log10, truth, cfg, path, z, label):
    model = crop_to_truth(model_log10, truth)
    label = label.replace('log10_mean_rho', 'E[rho]').replace('mean_rho', 'E[rho]')
    label = label.replace('mean_log10', '10 ** E[log10 rho]').replace('median_log10', 'Median rho')
    depth = sum(cfg.dz_list[:z]) + 0.5 * cfg.dz_list[z]
    (fig, axes) = plt.subplots(1, 3, figsize=(13, 4))
    for (ax, value, title) in zip(axes[:2], [truth[:, :, z], model[:, :, z]], ['Truth', label]):
        im = ax.imshow(10 ** value, cmap='rainbow_r', norm=LogNorm(10 ** cfg.model_min, 10 ** cfg.model_max))
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Resistivity (Ohm m)')
    im = axes[2].imshow(model[:, :, z] - truth[:, :, z], cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    axes[2].set_title('Reconstruction - truth')
    fig.colorbar(im, ax=axes[2], shrink=0.8, extend='both', label='log10 resistivity difference')
    for ax in axes:
        ax.set_xlabel('truth-grid column')
        ax.set_ylabel('truth-grid row')
    fig.suptitle(f'{cfg.case} | layer index {z} | depth {depth:g} m | centered common domain')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def structural_truth_assessment(sample_path, truth, cfg):
    samples = np.load(sample_path, mmap_mode='r')
    try:
        if samples.ndim != 4 or len(samples) < 2:
            raise ValueError('structural assessment requires independent 3-D model samples')
        crop_to_truth(samples[0], truth)
        sy = (samples.shape[1] - truth.shape[0]) // 2
        sx = (samples.shape[2] - truth.shape[1]) // 2
        dz = np.asarray(cfg.dz_list, dtype=np.float64)
        depth = dz.cumsum() - 0.5 * dz
        weights = cfg.dx * cfg.dy * dz[None, None, :]
        threshold = cfg.prior_mean - 0.5
        true_mask = truth < threshold
        true_volume = float((true_mask * weights).sum())
        true_center = float((true_mask * weights * depth).sum() / true_volume) if true_volume else None
        volumes = []
        centers = []
        areas = []
        for first in range(0, len(samples), 16):
            model = np.asarray(samples[first:first + 16, sy:sy + truth.shape[0], sx:sx + truth.shape[1], :])
            if not np.isfinite(model).all():
                raise ValueError('nonfinite posterior sample in assessment')
            mask = model < threshold
            volume = (mask * weights).sum(axis=(1, 2, 3))
            centers.extend(np.divide((mask * weights * depth).sum(axis=(1, 2, 3)), volume, out=np.full_like(volume, np.nan), where=volume > 0).tolist())
            volumes.extend(volume.tolist())
            areas.append(mask.sum(axis=(1, 2)) * cfg.dx * cfg.dy)

        def summary(values, reference):
            a = np.asarray(values)
            a = a[np.isfinite(a)]
            q = np.quantile(a, [0.05, 0.5, 0.95]).tolist() if len(a) else None
            return {'truth': reference, 'q05_median_q95': q, 'truth_inside_ci90': bool(q[0] <= reference <= q[2]) if q is not None and reference is not None else None}
        return {'scope': 'fixed-truth structural diagnostics on the centered common domain; not repeated-simulation calibration', 'draws': len(samples), 'shape': list(truth.shape), 'threshold_ohm_m': 10.0 ** threshold, 'conductor_volume_m3': summary(volumes, true_volume), 'conductor_centroid_depth_m': summary(centers, true_center), 'centroid_conditioning': 'positive conductor volume', 'probability_no_conductor': float(np.mean(np.asarray(volumes) == 0)), 'conductor_area_by_layer_m2': {'truth': (true_mask.sum(axis=(0, 1)) * cfg.dx * cfg.dy).tolist(), 'q05_median_q95': np.quantile(np.concatenate(areas), [0.05, 0.5, 0.95], axis=0).tolist()}}
    finally:
        samples._mmap.close()

def assess_result(result_dir, cfg):
    result_dir = Path(result_dir).resolve()
    if result_dir.name == 'posterior':
        result_dir = result_dir.parent
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    (truth, metadata) = load_truth(cfg)
    folder = result_dir / 'truth_assessment'
    folder.mkdir(exist_ok=True)
    report = {'truth': metadata, 'uniform_background': reconstruction_metrics(np.full_like(truth, math.log10(cfg.initial_resistivity)), truth, cfg), 'posterior': {}, 'point_model': {}, 'spatial_coverage': {}, 'training_snapshots': [], 'coverage_scope': 'fixed-model spatial coverage, not simulation-based calibration'}
    posterior = result_dir / 'posterior'
    if (posterior / 'posterior_samples_log10rho.npy').exists():
        report['structural_functionals'] = structural_truth_assessment(posterior / 'posterior_samples_log10rho.npy', truth, cfg)
    (selected, label) = (None, '')
    map_path = result_dir / 'map' / 'model_log10.npy'
    if (result_dir / 'map_refined' / 'model_log10.npy').exists():
        map_path = result_dir / 'map_refined' / 'model_log10.npy'
    if map_path.exists():
        (selected, label) = (np.load(map_path), 'MAP structure')
        report['point_model'] = reconstruction_metrics(selected, truth, cfg)
    for name in ['mean_rho', 'median_log10', 'mean_log10']:
        path = posterior / (name + '.npy')
        if path.exists():
            model = np.load(path)
            if name == 'mean_rho':
                if np.any(model <= 0):
                    raise ValueError('电阻率均值必须为正')
                model = np.log10(np.asarray(model, dtype=np.float64))
            report['posterior'][name] = reconstruction_metrics(model, truth, cfg)
            (selected, label) = (model, name)
    for (level, low, high) in [('90', 'q05', 'q95'), ('95', 'q025', 'q975')]:
        bounds = []
        for name in [low, high]:
            (log_path, rho_path) = (posterior / (name + '_log10.npy'), posterior / (name + '_rho.npy'))
            if log_path.exists():
                bounds.append(np.load(log_path))
            elif rho_path.exists():
                bounds.append(np.log10(np.load(rho_path)))
        if len(bounds) == 2:
            report['spatial_coverage'][level] = spatial_coverage(*bounds, truth, cfg)
    snapshots = []
    for (pattern, space) in [('representative_mean_rho_step_*.npy', 'log10_mean_rho'), ('representative_mean_log10_step_*.npy', 'mean_log10')]:
        for path in result_dir.glob(pattern):
            step = int(path.stem.rsplit('_', 1)[1])
            snapshots.append((step, path, space))
    for (step, path, space) in sorted(snapshots, key=lambda item: (item[0], item[2])):
        try:
            model = np.load(path)
        except (ValueError, OSError, EOFError):
            continue
        if space == 'log10_mean_rho':
            model = np.log10(np.asarray(model, dtype=np.float64))
        row = {'step': step, 'estimator': space, **reconstruction_metrics(model, truth, cfg)}
        report['training_snapshots'].append(row)
        if not report['posterior']:
            (selected, label) = (model, f'{space}, step {step}')
    if selected is None:
        raise FileNotFoundError('该目录没有可评价的点模型、后验统计或中间模型')
    anomaly_counts = (np.abs(truth - truth_background(truth)) > 0.5).sum(axis=(0, 1))
    for z in sorted({cfg.z_show, int(np.argmax(anomaly_counts))}):
        plot_truth_comparison(selected, truth, cfg, folder / f'truth_comparison_z{z:02d}.png', z, label)
    (folder / 'truth_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if report['training_snapshots']:
        keys = ['step', 'estimator', 'rmse_log10_all', 'rmse_log10_anomaly', 'rmse_log10_background', 'anomaly_iou', 'conductor_iou', 'resistor_iou', 'correlation', 'ssim_log10_volume_3d', 'ssim_log10_all_layers', 'ssim_log10_anomaly_layers']
        with (folder / 'reconstruction_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['training_snapshots'])
    print('[Truth assessment]', folder, flush=True)
    return report

@torch.no_grad()
def save_training_result(flow, device, cfg, out, step):
    folder = out / 'training_slices'
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 300003)
    rho_sum = torch.zeros(flow.shape3, dtype=torch.float64, device=device)
    log10_sum = torch.zeros_like(rho_sum)
    n = cfg.plot_representative_draws
    was_training = flow.training
    flow.eval()
    try:
        for start in range(0, n, cfg.plot_representative_batch):
            count = min(cfg.plot_representative_batch, n - start)
            (models, _) = flow.sample_and_log_prob(count, device, antithetic=False, generator=generator)
            rho_sum += torch.pow(10.0, models[:, 0].double()).sum(dim=0)
            log10_sum += models[:, 0].double().sum(dim=0)
    finally:
        flow.train(was_training)
    rho_mean = (rho_sum / n).cpu().numpy().astype(np.float32)
    np.save(out / f'representative_mean_rho_step_{step:06d}.npy', rho_mean)
    mean_log10 = (log10_sum / n).cpu().numpy().astype(np.float32)
    np.save(out / f'representative_mean_log10_step_{step:06d}.npy', mean_log10)
    geometric_mean = 10.0 ** mean_log10
    stem = f'representative_geometric_mean_rho_step_{step:06d}'
    np.save(out / (stem + '.npy'), geometric_mean)
    write_vtr(geometric_mean, out / (stem + '.vtr'), cfg, 'geometric_mean_resistivity')
    z = cfg.z_show
    depth = sum(cfg.dz_list[:z]) + 0.5 * cfg.dz_list[z]
    (fig, ax) = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rho_mean[:, :, z], cmap='Spectral', norm=LogNorm(vmin=3, vmax=750))
    fig.colorbar(im, ax=ax, extend='both', label='Resistivity (Ω·m)')
    ax.set_title(f'{cfg.case} E[rho] | step {step} | depth {depth:g} m')
    ax.set_xlabel('model column')
    ax.set_ylabel('model row')
    fig.tight_layout()
    figure_path = folder / f'step_{step:06d}.png'
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print('[Result]', step, figure_path, flush=True)
    if cfg.truth_path:
        (truth, metadata) = load_truth(cfg)
        metric = {'step': step, 'estimator': 'mean_log10', 'truth': metadata, **reconstruction_metrics(mean_log10, truth, cfg)}
        metric['posterior_arithmetic_mean'] = {'estimator': 'log10(E[rho])', 'representative_draws': n, **reconstruction_metrics(np.log10(np.asarray(rho_mean, dtype=np.float64)), truth, cfg)}
        (folder / f'truth_metrics_{step:06d}.json').write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding='utf-8')
        plot_truth_comparison(mean_log10, truth, cfg, folder / f'truth_step_{step:06d}.png', cfg.z_show, '10^E[log10 rho]')

def plot_stats(stats, cfg, out):
    z = cfg.z_show
    depth = sum(cfg.dz_list[:z]) + 0.5 * cfg.dz_list[z]
    (fig, axes) = plt.subplots(1, 4, figsize=(16, 4.2))
    arrays = [stats['mean_rho'], stats['median_rho'], stats['std_rho'], stats['ci90_width_rho']]
    titles = ['Mean resistivity', 'Median resistivity', 'Resistivity std', '90% interval width']
    for (i, (ax, a, title)) in enumerate(zip(axes, arrays, titles)):
        if i < 2:
            im = ax.imshow(a[:, :, z], cmap='Spectral', norm=LogNorm(3, 750))
        else:
            im = ax.imshow(a[:, :, z], cmap='magma', vmin=0, vmax=float(np.quantile(a, 0.995)))
        ax.set_title(title)
        ax.set_xlabel('model column')
        ax.set_ylabel('model row')
        fig.colorbar(im, ax=ax, shrink=0.8, extend='both' if i < 2 else 'max', label='Ω·m')
    fig.suptitle(f'{cfg.case} | depth {depth:g} m | pointwise marginal summaries')
    fig.tight_layout()
    fig.savefig(out / 'posterior_summary.png', dpi=180)
    plt.close(fig)

def posterior_statistics(samples, voxel_chunk=4096):
    shape = samples.shape[1:]
    flat = samples.reshape(len(samples), -1)
    keys = ['mean_log10', 'median_log10', 'std_log10', 'mean_rho', 'median_rho', 'std_rho', 'q025_log10', 'q05_log10', 'q95_log10', 'q975_log10', 'q025_rho', 'q05_rho', 'q95_rho', 'q975_rho']
    stats = {k: np.empty(flat.shape[1], np.float32) for k in keys}
    for start in range(0, flat.shape[1], voxel_chunk):
        sl = slice(start, start + voxel_chunk)
        a = np.asarray(flat[:, sl], dtype=np.float64)
        rho = 10 ** a
        for (suffix, values) in [('log10', a), ('rho', rho)]:
            stats['mean_' + suffix][sl] = values.mean(0)
            stats['std_' + suffix][sl] = values.std(0, ddof=1 if len(samples) > 1 else 0)
            q = np.quantile(values, [0.025, 0.05, 0.5, 0.95, 0.975], axis=0)
            for (name, arr) in zip(['q025', 'q05', 'median', 'q95', 'q975'], q):
                stats[name + '_' + suffix][sl] = arr
    stats = {k: v.reshape(shape) for (k, v) in stats.items()}
    stats['geometric_mean_rho'] = 10.0 ** stats['mean_log10']
    stats['ci90_width_rho'] = stats['q95_rho'] - stats['q05_rho']
    stats['ci95_width_rho'] = stats['q975_rho'] - stats['q025_rho']
    return stats

def write_vtr(volume, path, cfg, name='resistivity'):
    if not cfg.export_vtr:
        return
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    (ny, nx, nz) = volume.shape
    extent = f'0 {nx} 0 {ny} 0 {nz}'
    root = Element('VTKFile', type='RectilinearGrid', version='0.1', byte_order='LittleEndian')
    grid = SubElement(root, 'RectilinearGrid', WholeExtent=extent)
    piece = SubElement(grid, 'Piece', Extent=extent)
    cell = SubElement(piece, 'CellData', Scalars=name)
    field = SubElement(cell, 'DataArray', type='Float32', Name=name, format='ascii')
    field.text = ' '.join((f'{v:.7g}' for v in volume.transpose(2, 0, 1).ravel()))
    coord = SubElement(piece, 'Coordinates')
    for (key, a) in [('X', np.arange(nx + 1) * cfg.dx), ('Y', np.arange(ny + 1) * cfg.dy), ('Z', np.r_[0, np.cumsum(cfg.dz_list)])]:
        item = SubElement(coord, 'DataArray', type='Float64', Name=key, format='ascii')
        item.text = ' '.join((f'{v:.10g}' for v in a))
    ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

def save_structural_probabilities(samples, cfg, folder):
    n = len(samples)
    shape = samples.shape[1:]
    p_low = np.zeros(shape, np.float64)
    p_high = np.zeros(shape, np.float64)
    dz = np.asarray(cfg.dz_list, dtype=np.float64)
    depths = dz.cumsum() - 0.5 * dz
    volume_weights = cfg.dx * cfg.dy * dz[None, None, :]
    volumes = []
    centroids = []
    areas = []
    conductance = []
    low = cfg.prior_mean - 0.5
    high = cfg.prior_mean + 0.5
    for start in range(0, n, 16):
        a = np.asarray(samples[start:start + 16], dtype=np.float64)
        conductor = a < low
        resistor = a > high
        p_low += conductor.sum(0)
        p_high += resistor.sum(0)
        weight = conductor * volume_weights
        v = weight.sum(axis=(1, 2, 3))
        numerator = (weight * depths[None, None, None, :]).sum(axis=(1, 2, 3))
        volumes.extend(v.tolist())
        centroids.extend(np.divide(numerator, v, out=np.full_like(v, np.nan), where=v > 0).tolist())
        areas.append(conductor.sum(axis=(1, 2)) * cfg.dx * cfg.dy)
        conductance.append((10.0 ** (-a) * dz[None, None, None, :]).sum(-1))
    for (name, values) in [('probability_conductor', p_low / n), ('probability_resistor', p_high / n)]:
        values = values.astype(np.float32)
        np.save(folder / (name + '.npy'), values)
        write_vtr(values, folder / (name + '.vtr'), cfg, name)
    column = np.concatenate(conductance)
    area = np.concatenate(areas)
    np.savez_compressed(folder / 'structural_functionals_per_draw.npz', draw_index=np.arange(n, dtype=np.int64), conductor_volume_m3=np.asarray(volumes, dtype=np.float64), conductor_centroid_depth_m=np.asarray(centroids, dtype=np.float64), conductor_area_by_layer_m2=area, mean_column_conductance_s=column.mean(axis=(1, 2)), horizontal_integrated_conductance_s_m2=column.sum(axis=(1, 2)) * cfg.dx * cfg.dy, conductor_threshold_ohm_m=np.asarray(10.0 ** low), layer_center_depth_m=depths)
    for (name, values) in [('mean', column.mean(0)), ('q05', np.quantile(column, 0.05, axis=0)), ('q95', np.quantile(column, 0.95, axis=0))]:
        np.save(folder / ('column_conductance_' + name + '.npy'), values.astype(np.float32))
    finite_centroids = np.asarray(centroids)[np.isfinite(centroids)]
    report = {'scope': 'events under the fitted q on the complete model grid; not calibrated probabilities by construction', 'conductor_threshold_ohm_m': 10.0 ** low, 'resistor_threshold_ohm_m': 10.0 ** high, 'grid_shape': list(shape), 'draws': n, 'quantiles': [0.05, 0.5, 0.95], 'conductor_volume_m3': np.quantile(volumes, [0.05, 0.5, 0.95]).tolist(), 'probability_no_conductor': float(np.mean(np.asarray(volumes) == 0)), 'conductor_volume_centroid_depth_m': np.quantile(finite_centroids, [0.05, 0.5, 0.95]).tolist() if len(finite_centroids) else None, 'centroid_conditioning': 'conditional on a positive conductor volume', 'conductor_area_by_layer_m2': np.quantile(area, [0.05, 0.5, 0.95], axis=0).tolist(), 'layer_center_depth_m': depths.tolist(), 'column_conductance_unit': 'siemens'}
    (folder / 'structural_probabilities.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

@torch.no_grad()
def finish(flow, problem, cfg, out, selected_step):
    from cv_export import finish_cv
    return finish_cv(flow, problem, cfg, out, selected_step, globals())

def config_from_checkpoint(payload):
    config = dict(payload['config'])
    for key in ('initial_model_path', 'coarse_checkpoint_path', 'warmstart_flow_checkpoint_path', 'result_dir'):
        old_path = str(config.get(key, '')).replace('\\', '/')
        if '/程序/' in old_path and '/result/' in old_path:
            config[key] = str(ROOT / 'result' / old_path.split('/result/', 1)[1])
    if config.get('checkpoint_path') in ('kzy/HDF_coord_True_final_epoch_200_30m_spacing.pkl', '../AEM_Diffusion_3D_Inv_Orkdalen/kzy/HDF_coord_True_final_epoch_200_30m_spacing.pkl'):
        config['checkpoint_path'] = Config().checkpoint_path
    config.setdefault('initial_resistivity', 10.0 ** config.get('prior_mean', 2.0))
    if 'posterior_family' not in config:
        config.update(posterior_family='control', map_refine_steps=0, sensitivity_probes=0)
    if payload.get('format') == 'vi_v8':
        config.update(horizontal_coarsening=1, likelihood_mode='gaussian', initial_model_path='', map_steps=0, map_only=False)
    elif payload.get('format') not in ('vi_v8_structure', 'vi_v8_map', 'vi_v8_multiscale', 'vi_v8_refine'):
        raise ValueError('只支持v8断点，原v5/v7流结构不兼容')
    return Config(**config)

def model_logits(model, cfg):
    fraction = ((model - cfg.model_min) / (cfg.model_max - cfg.model_min)).clamp(1e-06, 1 - 1e-06)
    return torch.logit(fraction)

@torch.no_grad()
def rebase_flow_support(flow, original, cfg):
    if cfg.model_min > original.model_min or cfg.model_max < original.model_max:
        raise ValueError('support continuation may only widen the original bounds')
    new_bound = flow.bound
    flow.bound = BoundedTransform(original.model_min, original.model_max, flow.numel)
    zero = torch.zeros((1, *flow.base_shape), device=flow.global_affine.bias.device)
    (anchor, _) = flow.forward_from_base(zero)
    (previous_logits, _) = flow.bound.inverse(anchor)
    flow.bound = new_bound
    shift = haar_analysis(model_logits(anchor, cfg) - previous_logits)
    flow.global_affine.bias.add_(shift)
    (rebuilt, _) = flow.forward_from_base(zero)
    error = float((rebuilt - anchor).abs().max())
    if error > 0.0001:
        raise RuntimeError('support continuation failed to preserve its physical anchor')
    return error

def point_from_logits(logits, cfg):
    controls = cfg.model_min + (cfg.model_max - cfg.model_min) * torch.sigmoid(logits)
    return (resize_model(controls, (cfg.ny, cfg.nx, cfg.nz)), controls)

@torch.no_grad()
def evaluate_point(model, problem, cfg):
    result = predictive_summary(model[:, 0].detach().cpu().numpy(), problem)
    result['prior_energy'] = float(prior_energy(model, cfg)[0])
    result['point_objective'] = cfg.data_weight * result['expected_data_energy'] + result['prior_energy']
    return scalar_metrics(result)

def fit_map(problem, cfg, out, hashes, resume_payload=None):
    folder = out / 'map'
    folder.mkdir(exist_ok=True)
    objective_cfg = Config(**{**asdict(cfg), 'prior_mode': 'tv'}) if cfg.posterior_family == 'multiscale' else cfg
    shape = control_shape(cfg)
    initial = torch.full((1, 1, cfg.ny, cfg.nx, cfg.nz), math.log10(cfg.initial_resistivity), device=problem.device)
    if cfg.initial_model_path:
        path = (ROOT / cfg.initial_model_path).resolve()
        if cfg.truth_path and path == (ROOT / cfg.truth_path).resolve():
            raise ValueError('真值只用于评价，不能作为反演初值')
        a = np.load(path)
        if a.shape not in ((cfg.ny, cfg.nx, cfg.nz), shape) or not np.isfinite(a).all():
            raise ValueError('initial_model_path必须是有限的log10电阻率三维数组')
        if a.min() < cfg.model_min or a.max() > cfg.model_max:
            raise ValueError('初始模型超出支持域；请明确单位和模型范围')
        initial = torch.as_tensor(a, dtype=torch.float32, device=problem.device)[None, None]
    logits = nn.Parameter(model_logits(resize_model(initial, shape), cfg))
    optimizer = torch.optim.Adam([logits], lr=cfg.map_learning_rate)
    rng = np.random.default_rng(cfg.seed + 510007)
    history = []
    start = 1
    with torch.no_grad():
        (initial_fine, _) = point_from_logits(logits, cfg)
        best_metrics = evaluate_point(initial_fine, problem, objective_cfg)
        best_logits = logits.detach().clone()
        best_step = 0
        np.save(folder / 'initial_log10.npy', initial_fine[0, 0].cpu().numpy())
    if resume_payload is not None:
        if {k: v['sha256'] for (k, v) in hashes.items()} != {k: v['sha256'] for (k, v) in resume_payload['inputs'].items()}:
            raise ValueError('MAP断点输入文件已变化')
        saved = asdict(config_from_checkpoint(resume_payload))
        current = asdict(cfg)
        for key in ('result_dir', 'truth_path', 'z_show', 'map_only'):
            saved.pop(key, None)
            current.pop(key, None)
        if json.dumps(saved, sort_keys=True) != json.dumps(current, sort_keys=True):
            raise ValueError('MAP续训参数与断点不一致')
        with torch.no_grad():
            logits.copy_(resume_payload['logits'].to(problem.device))
        optimizer.load_state_dict(resume_payload['optimizer'])
        rng.bit_generator.state = resume_payload['rng_numpy']
        best_logits = resume_payload['best_logits'].to(problem.device)
        best_metrics = resume_payload['best_metrics']
        best_step = resume_payload['best_step']
        start = resume_payload['step'] + 1
        history = resume_payload['history']
    print('[MAP initial]', best_metrics['predictive_mean_rms'], best_metrics['point_objective'], flush=True)
    for step in range(start, cfg.map_steps + 1):
        tick = time.perf_counter()
        batch = rng.choice(problem.n_stations, min(cfg.map_station_batch_size, problem.n_stations), replace=False)
        optimizer.param_groups[0]['lr'] = cfg.map_learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, cfg.map_steps))))
        optimizer.zero_grad(set_to_none=True)
        (model, _) = point_from_logits(logits, cfg)
        (phi, gradient, rms, _) = streamed_gradient(model, batch, np.arange(5), problem, 1.0, problem.n_stations * 10)
        regularizer = prior_energy(model, objective_cfg).mean() / (problem.n_stations * 10)
        torch.autograd.backward((model, regularizer), (gradient, None))
        norm = torch.nn.utils.clip_grad_norm_([logits], cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'batch_rms': rms, 'point_objective_estimate': float(phi.mean()) + float(regularizer.detach()) * problem.n_stations * 10, 'full_objective': float('nan'), 'full_rms': float('nan'), 'seconds': time.perf_counter() - tick}
        del model, phi, gradient, regularizer
        if step % cfg.map_eval_every == 0 or step == cfg.map_steps:
            with torch.no_grad():
                (model, _) = point_from_logits(logits, cfg)
                ev = evaluate_point(model, problem, objective_cfg)
                row.update(full_objective=ev['point_objective'], full_rms=ev['predictive_mean_rms'])
                if ev['point_objective'] < best_metrics['point_objective']:
                    (best_metrics, best_logits, best_step) = (ev, logits.detach().clone(), step)
                np.save(folder / f'model_log10_step_{step:06d}.npy', model[0, 0].cpu().numpy())
            print('[MAP full]', step, ev['predictive_mean_rms'], ev['point_objective'], 'selected', best_step, flush=True)
        history.append(row)
        if step % cfg.map_eval_every == 0 or step == cfg.map_steps:
            atomic_save({'format': 'vi_v8_map', 'config': asdict(cfg), 'inputs': hashes, 'step': step, 'logits': logits.detach(), 'optimizer': optimizer.state_dict(), 'rng_numpy': rng.bit_generator.state, 'best_logits': best_logits, 'best_step': best_step, 'best_metrics': best_metrics, 'history': history}, out / 'map_last.pt')
            with (folder / 'history.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
        if step == start or step % 10 == 0:
            print('[MAP]', step, 'batch RMS', rms, 'grad', float(norm), flush=True)
    with torch.no_grad():
        (model, controls) = point_from_logits(best_logits, cfg)
        model_array = model[0, 0].cpu().numpy()
        np.save(folder / 'model_log10.npy', model_array)
        np.save(folder / 'model_rho.npy', 10.0 ** model_array)
        np.save(folder / 'controls_log10.npy', controls[0, 0].cpu().numpy())
        best_metrics.update(selected_step=best_step, control_shape=list(shape), selection='full data energy plus prior; no truth', initialization='external_model' if cfg.initial_model_path else 'uniform_halfspace', initial_resistivity=None if cfg.initial_model_path else cfg.initial_resistivity)
        best_metrics['training_step_seconds'] = sum((row['seconds'] for row in history))
        best_metrics['structure_prior_mode'] = objective_cfg.prior_mode
        best_metrics['timing_scope'] = 'MAP training steps only; excludes full evaluations, initial-model production and exports'
        if cfg.truth_path:
            (truth, metadata) = load_truth(cfg)
            best_metrics['truth'] = metadata
            best_metrics['reconstruction'] = reconstruction_metrics(model_array, truth, cfg)
            plot_truth_comparison(model_array, truth, cfg, folder / 'truth_comparison.png', cfg.z_show, 'MAP structure')
        (folder / 'metrics.json').write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        write_vtr(10.0 ** model_array, folder / 'model_rho.vtr', cfg, 'map_resistivity')
    return controls.detach()

def verified_uniform_coarse(cfg, hashes, out, device):
    path = (ROOT / cfg.coarse_checkpoint_path).resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    previous = config_from_checkpoint(payload)
    if payload.get('format') != 'vi_v8_map' or payload['step'] != previous.map_steps:
        raise ValueError('coarse_checkpoint_path requires a completed uniform MAP checkpoint')
    if previous.initial_model_path or cfg.initial_model_path:
        raise ValueError('reusable coarse stage must start from a uniform halfspace')
    if previous.prior_mode != 'tv' and previous.posterior_family != 'multiscale':
        raise ValueError('the shared coarse structure stage must use the declared TV objective')
    keys = ('case', 'seed', 'initial_resistivity', 'map_steps', 'map_learning_rate', 'map_station_batch_size', 'map_eval_every', 'ny', 'nx', 'nz', 'horizontal_coarsening', 'model_min', 'model_max', 'prior_mean', 'dx', 'dy', 'dz_list', 'likelihood_mode', 'laplace_scale', 'data_weight', 'relative_error', 'noise_floor', 'lambda0', 'tv_prior_strength', 'tv_axis_weights', 'tv_model_scale', 'tv_smoothing', 'grad_clip', 'response_polarity')
    for key in keys:
        if json.dumps(getattr(previous, key)) != json.dumps(getattr(cfg, key)):
            raise ValueError('coarse checkpoint differs in ' + key)
    if {k: v['sha256'] for (k, v) in hashes.items()} != {k: v['sha256'] for (k, v) in payload['inputs'].items()}:
        raise ValueError('coarse checkpoint observations, operator or index files differ')
    initial = np.load(path.parent / 'map' / 'initial_log10.npy')
    if not np.allclose(initial, math.log10(cfg.initial_resistivity), rtol=0, atol=1e-06):
        raise ValueError('saved coarse initialization is not the declared uniform halfspace')
    shutil.copytree(path.parent / 'map', out / 'map', dirs_exist_ok=True)
    shutil.copy2(path, out / 'map_last.pt')
    provenance = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'initial_resistivity': cfg.initial_resistivity, 'coarse_steps': payload['step'], 'coarse_training_seconds': sum((row['seconds'] for row in payload['history'])), 'timing_scope': 'prior coarse training must be included; full evaluation/export time is additional', 'validation': 'completed uniform stage, same objective, data, geometry, seed and optimizer budget'}
    (out / 'coarse_stage_reuse.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (_, controls) = point_from_logits(payload['best_logits'], cfg)
    print('[Coarse stage reused]', provenance, flush=True)
    return controls.detach()

def _train_impl(cfg, resume=''):
    from validate_protocol import validate_training_inputs
    validate_training_inputs(cfg)
    from probe_stage3 import require_passed_probe
    require_passed_probe(cfg)
    if not Path(cfg.result_dir).resolve().is_relative_to(JGR_ROOT.resolve()):
        raise ValueError('JGR field branch output must stay inside the separate JGR-SE directory')
    validate_config(cfg)
    if cfg.warmstart_flow_checkpoint_path and (cfg.initial_model_path or cfg.coarse_checkpoint_path or cfg.map_only):
        raise ValueError('warm-start posterior refit cannot combine with resume, external MAP, coarse reuse or map-only')
    if cfg.truth_path:
        load_truth(cfg)
    configure_runtime(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    out = Path(cfg.result_dir).resolve()
    if out.exists() and any(out.iterdir()) and (not resume):
        raise FileExistsError('output is not empty; use --output with a new directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'source_v8.py').exists():
        shutil.copy2(Path(__file__), out / 'source_v8.py')
    problem = Problem(cfg, device)
    warmstart_path = (ROOT / cfg.warmstart_flow_checkpoint_path).resolve() if cfg.warmstart_flow_checkpoint_path else None
    if warmstart_path:
        problem.paths['warmstart_flow'] = warmstart_path
    if cfg.initial_model_path:
        problem.paths['initial_model'] = (ROOT / cfg.initial_model_path).resolve()
    hashes = fingerprint(problem)
    resume_payload = torch.load(resume, map_location=device, weights_only=False) if resume else None
    multiscale = cfg.posterior_family == 'multiscale'
    metadata = {'config': asdict(cfg), 'inputs': hashes, 'python': sys.version, 'torch': torch.__version__, 'process_id': os.getpid(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu', 'flow_format': 'vi_v8_multiscale' if multiscale else 'vi_v8_structure', 'channel_names': CHANNEL_NAMES, 'posterior_coordinates': 'full fine-grid log10 resistivity' if multiscale else 'log10 resistivity on control grid', 'control_shape': list(control_shape(cfg)), 'output_shape': [cfg.ny, cfg.nx, cfg.nz], 'initialization': {'kind': 'previous_posterior_refit' if warmstart_path else 'external_model' if cfg.initial_model_path else 'uniform_halfspace', 'resistivity_ohm_m': None if cfg.initial_model_path or warmstart_path else cfg.initial_resistivity, 'external_model_path': cfg.initial_model_path or None, 'warmstart_flow_checkpoint_path': str(warmstart_path) if warmstart_path else None}}
    source_copy = out / ('source_v8_' + metadata['source_sha256'][:12] + '.py')
    if not source_copy.exists():
        shutil.copy2(Path(__file__), source_copy)
    map_controls = None
    diagonal = None
    map_model = None
    if resume_payload is None and (not warmstart_path) or (resume_payload is not None and resume_payload.get('format') in ('vi_v8_map', 'vi_v8_refine')):
        (out / 'runtime_config.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        print('[Initialization]', metadata['initialization'], flush=True)
        refining = resume_payload is not None and resume_payload.get('format') == 'vi_v8_refine'
        if refining:
            saved = asdict(config_from_checkpoint(resume_payload))
            current = asdict(cfg)
            for key in ('result_dir', 'truth_path', 'z_show', 'map_only'):
                saved.pop(key, None)
                current.pop(key, None)
            completed = resume_payload['step'] == cfg.map_refine_steps
            source_dir = Path(resume).resolve().parent
            if completed and source_dir != out:
                posterior_keys = ('global_transport_rank', 'n_coupling_layers', 'flow_hidden_channels', 'flow_dilations', 'mask_block_sizes', 'coordinate_conditioning', 'spline_bins', 'spline_tail_bound', 'spline_parameter_scale', 'precondition_strength_y', 'precondition_strength_x', 'precondition_strength_z', 'precondition_exponent', 'init_global_log_scale', 'antithetic_samples', 'num_steps', 'stage1_end_step', 'stage2_end_step', 'warmup_steps', 'beta_start', 'learning_rate', 'learning_rate_schedule_steps', 'location_lr_multiplier', 'scale_lr_multiplier', 'coupling_lr_multiplier', 'station_batch_size', 'frequencies_per_step', 'posterior_samples_per_step', 'log_every', 'plot_every', 'plot_representative_draws', 'plot_representative_batch', 'eval_every', 'eval_samples', 'checkpoint_every', 'n_posterior_samples', 'posterior_sample_batch', 'final_predictive_samples')
                for key in posterior_keys:
                    saved.pop(key, None)
                    current.pop(key, None)
            if json.dumps(saved, sort_keys=True) != json.dumps(current, sort_keys=True):
                raise ValueError('refinement resume configuration differs')
            if {k: v['sha256'] for (k, v) in hashes.items()} != {k: v['sha256'] for (k, v) in resume_payload['inputs'].items()}:
                raise ValueError('refinement input files changed')
            if completed and source_dir != out:
                for name in ('map', 'map_refined'):
                    if (source_dir / name).exists():
                        shutil.copytree(source_dir / name, out / name, dirs_exist_ok=True)
                for name in ('sensitivity.npz', 'sensitivity.json', 'structure_sensitivity.npz', 'structure_sensitivity.json', 'coarse_stage_reuse.json'):
                    if (source_dir / name).exists():
                        shutil.copy2(source_dir / name, out / name)
                (out / 'posterior_branch.json').write_text(json.dumps({'source_refinement': str(Path(resume).resolve()), 'source_sha256': hashlib.sha256(Path(resume).read_bytes()).hexdigest(), 'scope': 'new posterior fit after the same completed uniform-initialized structure stage; include shared stage cost'}, indent=2), encoding='utf-8')
            map_model = resume_payload['initial'].to(device)
            diagonal = resume_payload['diagonal'].to(device)
        elif cfg.coarse_checkpoint_path and resume_payload is None:
            map_controls = verified_uniform_coarse(cfg, hashes, out, device)
        elif cfg.map_steps > 0 or cfg.initial_model_path or cfg.map_only or multiscale or (math.log10(cfg.initial_resistivity) != cfg.prior_mean):
            map_controls = fit_map(problem, cfg, out, hashes, resume_payload)
        if multiscale:
            if map_model is None:
                map_model = resize_model(map_controls, (cfg.ny, cfg.nx, cfg.nz))
            if diagonal is None:
                diagonal = estimate_sensitivity(map_model, problem, cfg, out)
                for suffix in ('.npz', '.json'):
                    shutil.copy2(out / ('sensitivity' + suffix), out / ('structure_sensitivity' + suffix))
            if cfg.map_refine_steps:
                map_model = fit_refinement(map_model, diagonal, problem, cfg, out, hashes, resume_payload if refining else None)
        if cfg.map_only:
            print('[Done] MAP structure', out / ('map_refined' if multiscale and cfg.map_refine_steps else 'map'), flush=True)
            return
        resume_payload = None
    flow = (MultiscaleFlow3D(cfg, device) if multiscale else SpatialFlow3D(cfg, device)).to(device)
    if multiscale and map_model is not None:
        geometry_path = out / 'sensitivity.npz'
        geometry = np.load(geometry_path) if geometry_path.exists() else None
        anchor_sha256 = hashlib.sha256(map_model.detach().cpu().numpy().tobytes()).hexdigest()
        if geometry is None or 'projected_vjps' not in geometry or 'anchor_sha256' not in geometry or (str(geometry['anchor_sha256']) != anchor_sha256):
            if geometry is not None:
                geometry.close()
            diagonal = estimate_sensitivity(map_model, problem, cfg, out)
            geometry = np.load(geometry_path)
        sketch = torch.as_tensor(geometry['projected_vjps'], device=device)
        diagonal = torch.as_tensor(geometry['fisher_diagonal'], device=device)
        geometry.close()
        flow.initialize_from_model(map_model, diagonal, sketch)
        del sketch
    elif map_controls is not None:
        with torch.no_grad():
            flow.global_affine.bias.copy_(model_logits(map_controls, cfg))
    map_controls = None
    map_model = None
    diagonal = None
    if warmstart_path and resume_payload is None:
        warm = torch.load(warmstart_path, map_location='cpu', weights_only=False)
        if warm.get('format') != 'vi_v8_multiscale':
            raise ValueError('warm-start source must be a complete v8 multiscale-flow checkpoint')
        original = config_from_checkpoint(warm)
        augmentation_record = None
        if cfg.warmstart_boundary_augmentation:
            from continuation_ready import validate_prerequisites, compare_configs
            augmentation_record = validate_prerequisites(cfg)
            compare_configs(original, cfg)
        required = ('case', 'ny', 'nx', 'nz', 'dx', 'dy', 'dz_list', 'model_min', 'model_max', 'prior_mean', 'posterior_family', 'prior_mode', 'prior_phase_shifts', 'detail_prior_scale', 'detail_prior_df', 'tv_prior_strength', 'horizontal_curvature_strength', 'likelihood_mode', 'laplace_scale', 'relative_error', 'noise_floor', 'response_polarity', 'observation_transform', 'height_interpolation', 'outside_height_policy', 'n_coupling_layers', 'flow_hidden_channels', 'flow_dilations', 'mask_block_sizes', 'coordinate_conditioning', 'spline_bins', 'spline_tail_bound', 'spline_parameter_scale', 'global_transport_rank')
        compared = tuple((key for key in required if not (cfg.warmstart_response_convention_change and key == 'response_polarity') and (not (cfg.warmstart_likelihood_change and key == 'likelihood_mode')) and (not (cfg.warmstart_support_change and key in ('model_min', 'model_max'))) and (not (cfg.warmstart_boundary_augmentation and key == 'outside_height_policy'))))
        if any((getattr(original, key) != getattr(cfg, key) for key in compared)):
            raise ValueError('warm-start posterior geometry, prior, or likelihood differs from source')
        response_convention_changed = list(original.response_polarity) != list(cfg.response_polarity)
        if response_convention_changed:
            if not cfg.warmstart_response_convention_change:
                raise ValueError('Changing signed-response convention requires an explicit new-target branch')
            if list(original.response_polarity) != [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1] or list(cfg.response_polarity) != [1, 1, -1, -1, 1, 1, -1, -1, 1, 1]:
                raise ValueError('Only the independently validated, locked880-Re convention change is authorized')
        likelihood_changed = original.likelihood_mode != cfg.likelihood_mode
        if likelihood_changed and (not cfg.warmstart_likelihood_change):
            raise ValueError('changing the warm-start likelihood requires an explicit new-target branch')
        original_manifest = json.loads((ROOT / original.field_manifest_path).read_text(encoding='utf-8'))
        current_manifest = json.loads((ROOT / cfg.field_manifest_path).read_text(encoding='utf-8'))
        same_split = all((original_manifest['files'][part]['source_row_ids_zero_based'] == current_manifest['files'][part]['source_row_ids_zero_based'] for part in ('train', 'heldout')))
        fit_all = current_manifest.get('fit_all_after_separate_validation') and current_manifest['selected_station_count'] == current_manifest['source_station_count']
        if original_manifest['source_data_sha256'] != current_manifest['source_data_sha256'] or not (same_split or fit_all):
            raise ValueError('warm-start requires the same survey and either the same split or a final all-station fit')
        flow.load_state_dict(warm['flow'])
        if augmentation_record is not None:
            from continuation_ready import verify_same_base_mapping
            augmentation_record['initial_q_mapping_check'] = verify_same_base_mapping(flow, original, warm['flow'], cfg)
            (out / 'all3949_augmentation_provenance.json').write_text(json.dumps(augmentation_record, ensure_ascii=False, indent=2), encoding='utf-8')
        support_changed = original.model_min != cfg.model_min or original.model_max != cfg.model_max
        anchor_error = rebase_flow_support(flow, original, cfg) if support_changed else 0.0
        (out / 'warmstart_provenance.json').write_text(json.dumps({'source_checkpoint': str(warmstart_path), 'source_checkpoint_sha256': hashes['warmstart_flow']['sha256'], 'source_selected_step': warm['step'], 'source_training_station_count': original_manifest['files']['train']['count'], 'source_heldout_station_count': original_manifest['files']['heldout']['count'], 'final_fit_station_count': problem.n_stations, 'source_data_weight': original.data_weight, 'target_data_weight': cfg.data_weight, 'source_likelihood_mode': original.likelihood_mode, 'target_likelihood_mode': cfg.likelihood_mode, 'likelihood_changed': likelihood_changed, 'source_response_polarity': list(original.response_polarity), 'target_response_polarity': list(cfg.response_polarity), 'response_convention_changed': response_convention_changed, 'new_target_interpretation': 'The previous complete q is only a numerical initialization. The likelihood uses the newly locked P and raw signed observations exactly once. Old q is not multiplied into the prior. This all-data branch has no new independent holdout.', 'optimizer_reinitialized': True, 'source_log10_support': [original.model_min, original.model_max], 'target_log10_support': [cfg.model_min, cfg.model_max], 'support_changed': support_changed, 'zero_latent_anchor_max_abs_change': anchor_error, 'interpretation': 'This continuation fits all observations. In-sample residuals do not independently validate the current likelihood, bounds, or hyperparameters.' if fit_all else 'The original spatial train/heldout split is preserved; any target or hyperparameter changes still require separate validation.'}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[Warmstart flow] source', warmstart_path, 'stations', problem.n_stations, flush=True)
        del warm
    bias = [flow.global_affine.bias]
    scale = [flow.global_affine.raw_log_scale]
    if hasattr(flow, 'transport'):
        scale.append(flow.transport.raw_log_scale)
    layers = list(flow.layers.parameters())
    rates = [cfg.learning_rate * x for x in (cfg.location_lr_multiplier, cfg.scale_lr_multiplier, cfg.coupling_lr_multiplier)]
    opt = torch.optim.Adam([{'params': p, 'lr': lr} for (p, lr) in zip([bias, scale, layers], rates)])
    rng = np.random.default_rng(cfg.seed + 661)
    from cv_constraints import backward_cv_penalty, evaluate_cv
    cv_generator = torch.Generator(device=device).manual_seed(cfg.cv_seed)
    best_selection_key = (True, float('inf'))
    start = 1
    best = float('inf')
    elapsed = 0.0
    history = []
    if resume_payload is not None:
        payload = resume_payload
        if payload.get('format') not in ('vi_v8', 'vi_v8_structure', 'vi_v8_multiscale'):
            raise ValueError('legacy v5/v7 checkpoints are incompatible; start a new v8 run')
        saved = asdict(config_from_checkpoint(payload))
        current = asdict(cfg)
        fixed_schedule = cfg.learning_rate_schedule_steps
        if fixed_schedule > 0 and saved.get('learning_rate_schedule_steps') == fixed_schedule and (saved['num_steps'] <= cfg.num_steps <= fixed_schedule):
            if saved['num_steps'] != cfg.num_steps:
                metadata['budget_extension'] = {'from_steps': saved['num_steps'], 'to_steps': cfg.num_steps, 'unchanged_learning_rate_schedule_steps': fixed_schedule}
            saved.pop('num_steps')
            current.pop('num_steps')
        for key in ('result_dir', 'eval_every', 'log_every', 'checkpoint_every', 'plot_every', 'plot_representative_draws', 'plot_representative_batch', 'n_posterior_samples', 'final_predictive_samples', 'truth_path', 'z_show', 'cv_finalize'):
            saved.pop(key, None)
            current.pop(key, None)
        if json.dumps(saved, sort_keys=True) != json.dumps(current, sort_keys=True):
            raise ValueError('resume config differs from saved target, architecture or optimization schedule')
        if {k: v['sha256'] for (k, v) in hashes.items()} != {k: v['sha256'] for (k, v) in payload['inputs'].items()}:
            raise ValueError('input files changed since checkpoint')
        flow.load_state_dict(payload['flow'])
        opt.load_state_dict(payload['optimizer'])
        rng.bit_generator.state = payload['rng_numpy']
        torch.set_rng_state(payload['rng_torch'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in payload['rng_cuda']])
        random.setstate(payload['rng_python'])
        start = payload['step'] + 1
        best = payload['best_negative_elbo']
        elapsed = payload['elapsed_seconds']
        history = payload['history']
        cv_generator.set_state(payload['rng_cv'].cpu())
        best_selection_key = tuple(payload['best_cv_selection_key'])
        source_best = Path(resume).resolve().parent / 'best.pt'
        if source_best.exists() and source_best != out / 'best.pt':
            shutil.copy2(source_best, out / 'best.pt')
        elif not source_best.exists() and np.isfinite(best):
            raise FileNotFoundError('resume needs the best.pt saved beside last.pt')
    (out / 'runtime_config.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[Run]', cfg.case, device, 'stations', problem.n_stations, 'output', out, flush=True)
    if resume_payload is None:
        flow.set_active_coupling_layers(len(flow.layers))
        (diagnostic, initial_mean) = evaluate_cv(flow, device, cfg)
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(cfg.cv_seed + 1)
            zm = torch.randn((1, *flow.base_shape), device=device, generator=generator)
            (dm, dq) = flow.forward_from_base(zm)
            recovered = flow.inverse_to_base(dm)
            checkq = flow.log_prob(dm)
            diagnostic['density_roundtrip_base_rmse'] = float((recovered - zm).square().mean().sqrt())
            diagnostic['density_forward_inverse_logq_error_per_variable'] = float((dq - checkq).abs().max() / flow.numel)
        if not all((math.isfinite(diagnostic[k]) for k in ['density_roundtrip_base_rmse', 'density_forward_inverse_logq_error_per_variable'])):
            raise FloatingPointError('initial full-flow density roundtrip nonfinite')
        if diagnostic['density_forward_inverse_logq_error_per_variable'] > 0.01:
            raise FloatingPointError('initial full-flow density roundtrip inaccurate')
        diagnostic.update(interpretation='Complete training-MAP-initialized distribution before joint optimization of the original signed-response target and physical-CV soft objective; no calibration claim.', variables=flow.numel)
        (out / 'initial_cv_and_density.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[Initial independent CV]', json.dumps(diagnostic, ensure_ascii=False), flush=True)
        del zm, dm, dq, recovered, checkq
        (initial_ev, _) = evaluate(flow, problem, cfg)
        initial_ev.update(diagnostic)
        with torch.no_grad():
            initial_physical = torch.as_tensor(np.log10(initial_mean), device=device, dtype=torch.float32).view(1, 1, *flow.shape3)
            initial_response = problem.forward_models(initial_physical, np.arange(problem.n_stations))[0] + problem.bias
            initial_raw_rms = float((initial_response - problem.label).square().mean().sqrt())
        initial_ev['physical_mean_model_raw_rms_ppm'] = initial_raw_rms
        initial_ev['regularized_objective_per_variable'] = initial_ev['negative_elbo'] / flow.numel + diagnostic['cv_penalty_per_variable']
        feasible = diagnostic['cv_summary']['p95'] <= cfg.cv_acceptance_p95
        best_selection_key = (not feasible, initial_ev['regularized_objective_per_variable'])
        best = initial_ev['negative_elbo']
        initial_ev['cv_selection_key'] = list(best_selection_key)
        initial_ev['cv_selection_rule'] = 'Frozen: approximate physical-CV feasibility, then complete training J/D plus fixed CV penalty; no validation or raw-RMS checkpoint selection.'
        (out / 'baseline_new_target_evaluation.json').write_text(json.dumps(scalar_metrics(initial_ev), ensure_ascii=False, indent=2), encoding='utf-8')
        np.savez_compressed(out / 'baseline_new_target_evaluation_responses.npz', mean_rho=initial_response.cpu().numpy(), observations=problem.label.cpu().numpy())
        atomic_save({'format': 'vi_v8_multiscale' if multiscale else 'vi_v8_structure', 'step': 0, 'flow': flow.state_dict(), 'optimizer': opt.state_dict(), 'config': asdict(cfg), 'inputs': hashes, 'rng_numpy': rng.bit_generator.state, 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], 'rng_python': random.getstate(), 'best_negative_elbo': best, 'history': [], 'elapsed_seconds': 0.0, 'rng_cv': cv_generator.get_state(), 'best_cv_selection_key': list(best_selection_key), 'additional_variational_regularizer': 'unchanged physical-rho-CV penalty; new signed-response target, no calibration claim'}, out / 'best.pt')
        shutil.copy2(out / 'best.pt', out / 'initial_q.pt')
        print('[Initial NEW-target candidate]', json.dumps(scalar_metrics(initial_ev)), flush=True)
        del initial_mean, initial_physical, initial_response
    t0 = time.perf_counter()
    active = None
    for step in range(start, cfg.num_steps + 1):
        tick = time.perf_counter()
        stage = phase(step, cfg)
        if stage != active:
            set_phase(flow, stage)
            active = stage
            print('[Stage]', stage, 'step', step, flush=True)
        si = rng.choice(problem.n_stations, min(cfg.station_batch_size, problem.n_stations), replace=False)
        fi = np.sort(rng.choice(5, cfg.frequencies_per_step, replace=False))
        n_mc = 1 if stage == 1 else cfg.posterior_samples_per_step
        for (i, (lr, activation)) in enumerate(zip(rates, [0, cfg.stage1_end_step, cfg.stage2_end_step])):
            decay_steps = cfg.learning_rate_schedule_steps or cfg.num_steps
            frac = max(0, step - activation) / max(1, decay_steps - activation)
            opt.param_groups[i]['lr'] = lr * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * min(1, frac))))
        opt.zero_grad(set_to_none=True)
        (m, q) = flow.sample_and_log_prob(n_mc, device)
        beta = cfg.beta_start + (1 - cfg.beta_start) * min(1, step / max(1, cfg.warmup_steps))
        (phi, grad, rms, expected_rms) = streamed_gradient(m, si, fi, problem, beta, flow.numel)
        prior = prior_energy(m, cfg)
        reg = (prior + q).mean() / flow.numel
        if not torch.isfinite(reg):
            raise FloatingPointError('nonfinite prior or flow density')
        torch.autograd.backward((m, reg), (grad, None))
        cv_update = None
        if step == 1 or step % cfg.cv_every == 0:
            cv_update = backward_cv_penalty(flow, device, cfg, cv_generator, period_multiplier=1.0 if step == 1 else cfg.cv_every)
            print('[CV gradient]', step, json.dumps(cv_update, ensure_ascii=False), flush=True)
        norm = torch.nn.utils.clip_grad_norm_([p for p in flow.parameters() if p.requires_grad], cfg.grad_clip, error_if_nonfinite=True)
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        row = {'step': step, 'stage': stage, 'beta': beta, 'negative_elbo': float((phi + prior.detach() + q.detach()).mean()), 'annealed_objective': float((beta * phi + prior.detach() + q.detach()).mean()), 'phi': float(phi.mean()), 'prior': float(prior.mean().detach()), 'logq': float(q.mean().detach()), 'batch_expected_sample_rms': expected_rms, 'batch_predictive_mean_rms': rms, 'grad_norm': float(norm), 'step_seconds': time.perf_counter() - tick, 'n_mc': n_mc, 'frequency_indices': ' '.join(map(str, fi)), 'full_negative_elbo': float('nan'), 'full_expected_sample_rms': float('nan'), 'full_predictive_mean_rms': float('nan'), 'elapsed_seconds': elapsed + time.perf_counter() - t0}
        row.update(cv_update_penalty=float('nan') if cv_update is None else cv_update['cv_penalty_per_variable'], cv_update_p50=float('nan') if cv_update is None else cv_update['cv_summary']['p50'], cv_update_p95=float('nan') if cv_update is None else cv_update['cv_summary']['p95'], full_cv_penalty=float('nan'), full_cv_p50=float('nan'), full_cv_p95=float('nan'), full_regularized_objective_per_variable=float('nan'), physical_mean_model_raw_rms_ppm=float('nan'))
        del m, q, phi, prior, reg, grad
        if cfg.plot_every > 0 and (step == 1 or step % cfg.plot_every == 0 or step == cfg.num_steps):
            save_training_result(flow, device, cfg, out, step)
        improved = False
        if step == cfg.num_steps or (cfg.eval_every > 0 and step % cfg.eval_every == 0):
            (ev, _) = evaluate(flow, problem, cfg)
            (cv_ev, mean_rho) = evaluate_cv(flow, device, cfg)
            ev.update(cv_ev)
            with torch.no_grad():
                physical_mean_model = torch.as_tensor(np.log10(mean_rho), device=device, dtype=torch.float32).view(1, 1, *flow.shape3)
                mean_prediction = problem.forward_models(physical_mean_model, np.arange(problem.n_stations))[0] + problem.bias
                raw_rms = float((mean_prediction - problem.label).square().mean().sqrt())
            ev['physical_mean_model_raw_rms_ppm'] = raw_rms
            ev['regularized_objective_per_variable'] = ev['negative_elbo'] / flow.numel + cv_ev['cv_penalty_per_variable']
            feasible = cv_ev['cv_summary']['p95'] <= cfg.cv_acceptance_p95
            selection_key = (not feasible, ev['regularized_objective_per_variable'])
            ev['cv_selection_key'] = list(selection_key)
            ev['cv_selection_rule'] = 'Lexicographic approximate CV feasibility (independent screening p95<=declared threshold), then original J/D plus declared physical-CV penalty. No heldout test or truth used.'
            row.update(full_negative_elbo=ev['negative_elbo'], full_expected_sample_rms=ev['expected_sample_rms'], full_predictive_mean_rms=ev['predictive_mean_rms'], full_cv_penalty=cv_ev['cv_penalty_per_variable'], full_cv_p50=cv_ev['cv_summary']['p50'], full_cv_p95=cv_ev['cv_summary']['p95'], full_regularized_objective_per_variable=ev['regularized_objective_per_variable'], physical_mean_model_raw_rms_ppm=raw_rms)
            np.save(out / f'evaluation_{step:06d}_mean_rho.npy', mean_rho.astype(np.float32))
            np.savez_compressed(out / f'evaluation_{step:06d}_mean_model_responses.npz', mean_rho=mean_prediction.cpu().numpy(), observations=problem.label.cpu().numpy())
            (out / f'evaluation_{step:06d}.json').write_text(json.dumps(scalar_metrics(ev), ensure_ascii=False, indent=2), encoding='utf-8')
            if stage == 3 and selection_key < best_selection_key:
                best = ev['negative_elbo']
                best_selection_key = selection_key
                improved = True
            print('[CV full evaluation]', step, 'CV p50/p95', cv_ev['cv_summary']['p50'], cv_ev['cv_summary']['p95'], 'F(meanrho) ppm', raw_rms, 'J/D', ev['negative_elbo'] / flow.numel, 'penalty', cv_ev['cv_penalty_per_variable'], flush=True)
            print('[Full predictive]', step, ev['expected_sample_rms'], ev['predictive_mean_rms'], flush=True)
        row['elapsed_seconds'] = elapsed + time.perf_counter() - t0
        history.append(row)
        if step % cfg.log_every == 0 or step == start:
            print(f"[VI] {step}/{cfg.num_steps} stage={stage} E-rms={row['batch_expected_sample_rms']:.3f} mean-rms={rms:.3f} J={row['negative_elbo']:.0f} grad={float(norm):.3g} train={row['step_seconds']:.2f}s", flush=True)
        if improved or step % cfg.checkpoint_every == 0 or step == cfg.num_steps:
            payload = {'format': 'vi_v8_multiscale' if multiscale else 'vi_v8_structure', 'step': step, 'flow': flow.state_dict(), 'optimizer': opt.state_dict(), 'config': asdict(cfg), 'inputs': hashes, 'rng_numpy': rng.bit_generator.state, 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], 'rng_python': random.getstate(), 'best_negative_elbo': best, 'history': history, 'elapsed_seconds': elapsed + time.perf_counter() - t0, 'rng_cv': cv_generator.get_state(), 'best_cv_selection_key': list(best_selection_key), 'additional_variational_regularizer': 'physical-rho-CV upper moment penalty; not calibrated v2 posterior'}
            if improved:
                atomic_save(payload, out / 'best.pt')
            atomic_save(payload, out / 'last.pt')
            if cfg.save_numbered_checkpoints and (step % cfg.checkpoint_every == 0 or step == cfg.num_steps):
                atomic_save(payload, out / f'checkpoint_{step:06d}.pt')
        if step % cfg.log_every == 0 or step == cfg.num_steps:
            with (out / 'vi_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys((key for entry in history for key in entry))))
                writer.writeheader()
                writer.writerows(history)
    if not cfg.cv_finalize:
        print('[Joint VI optimization completed] complete state saved; train-only predictive audit follows separately', flush=True)
        return
    selected = torch.load(out / 'best.pt', map_location=device, weights_only=False)
    flow.load_state_dict(selected['flow'])
    flow.set_active_coupling_layers(len(flow.layers))
    flow.eval()
    finish(flow, problem, cfg, out, selected['step'])
    print('[Done] best explicitly CV-constrained selection checkpoint', selected['step'], 'results', out, flush=True)

def train(cfg, resume=''):
    from datetime import datetime, timezone
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    destination = Path(cfg.result_dir).resolve()
    previously_nonempty = destination.exists() and any(destination.iterdir())
    status = 'failed'
    failure = None
    try:
        result = _train_impl(cfg, resume)
        status = 'completed'
        return result
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        destination = Path(cfg.result_dir).resolve()
        if destination.is_relative_to(JGR_ROOT.resolve()) and destination.exists() and (not previously_nonempty or bool(resume)):
            timing = {'status': status, 'started_utc': stamp, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'whole_invocation_seconds': time.perf_counter() - started, 'scope': 'one original-target cold-all3949 joint-VI invocation including operator/cold-DCT-MAP loading, fresh all-data sensitivity, joint updates and full-data/CV evaluations; cold MAP cost tracked separately; excludes later final export, independent audit/replay and offline operator training', 'failure': failure, 'GPU_peak_allocated_bytes': None, 'GPU_peak_reserved_bytes': None}
            if torch.cuda.is_initialized():
                timing['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                timing['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                timing['GPU_name'] = torch.cuda.get_device_name()
            timing_path = destination / 'execution_timing_cv_v3.json'
            if timing_path.exists():
                timing_path = destination / ('execution_timing_cv_v3_resume_' + str(time.time_ns()) + '.json')
            timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding='utf-8')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='可选的 JSON 参数文件；默认使用本文件顶部 Config')
    parser.add_argument('--output', help='指定新的结果目录')
    parser.add_argument('--resume', default='', help='从本算例 v8 的 last.pt 继续训练')
    parser.add_argument('--assess-result', help='对已有结果做 .rho 真值对照，不运行训练')
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument('--initial-model', help='热启动对照：由观测反演得到的log10电阻率模型.npy')
    initial_group.add_argument('--halfspace', type=float, nargs='?', const=100.0, help='从均匀半空间新起算，单位Ω·m；省略数值时为100，清除外部模型初值')
    parser.add_argument('--map-only', action='store_true', help='只运行结构恢复阶段并保存三维点模型')
    parser.add_argument('--evaluate-field', action='store_true', help='完整训练结束后评价空间留出点并生成水平、垂直剖面')
    parser.add_argument('--coarse-checkpoint', help='复用已完成且输入一致的均匀起算MAP阶段，成本一并记录')
    args = parser.parse_args(argv)
    if args.resume and (args.halfspace is not None or args.initial_model):
        parser.error('续训沿用原初始化；改变初值请新起算，不与--resume同时使用')
    if args.coarse_checkpoint and (args.resume or args.initial_model):
        parser.error('--coarse-checkpoint用于均匀起算的新分支，不与续训或外部模型初值混用')
    cfg = Config()
    if not args.config and (not args.resume) and (not args.assess_result):
        for (key, value) in json.loads((ROOT / 'config_cv20.json').read_text('utf-8')).items():
            setattr(cfg, key, value)
    if args.assess_result:
        result_path = Path(args.assess_result).resolve()
        if result_path.name == 'posterior':
            result_path = result_path.parent
        metadata_path = result_path / 'runtime_config.json'
        if metadata_path.exists():
            saved_cfg = json.loads(metadata_path.read_text(encoding='utf-8-sig'))['config']
            if saved_cfg['case'] != cfg.case:
                raise ValueError('请使用对应算例的 v8 程序进行真值对照')
            for key in ('ny', 'nx', 'nz', 'dx', 'dy', 'dz_list', 'model_min', 'model_max', 'prior_mean'):
                setattr(cfg, key, saved_cfg[key])
    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        cfg = config_from_checkpoint(payload)
    if args.config:
        values = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        unknown = set(values) - {field.name for field in fields(Config)}
        if unknown:
            raise ValueError('未知配置字段：' + str(unknown))
        for (key, value) in values.items():
            setattr(cfg, key, value)
    if args.assess_result:
        assess_result(args.assess_result, cfg)
        return
    if args.initial_model:
        cfg.initial_model_path = args.initial_model
    if args.halfspace is not None:
        cfg.initial_model_path = ''
        cfg.initial_resistivity = args.halfspace
    if args.map_only:
        cfg.map_only = True
    if args.coarse_checkpoint:
        cfg.coarse_checkpoint_path = args.coarse_checkpoint
    if args.output:
        cfg.result_dir = args.output
    elif not cfg.result_dir:
        cfg.result_dir = str(ROOT / 'result' / f"v8_{time.strftime('%Y%m%d_%H%M%S')}")
    train(cfg, args.resume)
    if args.evaluate_field and (not cfg.map_only):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from evaluate_field import evaluate
        evaluate(Path(cfg.result_dir))
if __name__ == '__main__':
    main()
