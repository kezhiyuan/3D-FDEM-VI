from __future__ import annotations
import numpy as np
SYSTEM_POLARITY = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1], dtype=float)

def transform_observations(signed, mode='absolute_value'):
    data = np.asarray(signed)
    if data.shape[-1] != 10 or not np.isfinite(data).all():
        raise ValueError('Expected finite signed real/imaginary values in ten channels')
    if mode == 'absolute_value':
        return np.abs(data)
    if mode == 'signed':
        return np.array(data, copy=True)
    raise ValueError('Observation transform must be absolute_value or signed')

def prepare_heights(actual, mode='nearest', outside_policy='inflate'):
    actual = np.asarray(actual, dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError('Flight heights must be finite')
    if mode == 'nearest':
        queried = np.clip(np.round(actual / 10.0) * 10.0, 30.0, 90.0)
        excess = np.maximum(np.abs(actual - queried) - 5.0, 0.0)
        admissible = (actual >= 25) & (actual <= 95)
    elif mode == 'linear':
        queried = np.clip(actual, 30.0, 90.0)
        excess = np.abs(actual - queried)
        admissible = (actual >= 30) & (actual <= 90)
    else:
        raise ValueError('Height interpolation must be nearest or linear')
    if outside_policy == 'reject' and (not admissible.all()):
        raise ValueError('Height outside the explicitly selected inference range')
    if outside_policy not in ('reject', 'exclude', 'inflate'):
        raise ValueError('Outside-height policy must be reject, exclude, or inflate')
    return (queried, excess, admissible)

def select_height_response(outputs, height_indices, mode='nearest'):
    import torch
    if outputs.ndim != 4 or outputs.shape[2:] != (7, 2):
        raise ValueError('Expected (batch,frequency,7,2) operator outputs')
    h = height_indices.to(device=outputs.device)
    if h.shape != (outputs.shape[0],) or not torch.isfinite(h).all() or torch.any((h < 0) | (h > 6)):
        raise ValueError('Height index must be finite and lie within [0,6]')
    if mode == 'nearest':
        if not torch.equal(h, h.round()):
            raise ValueError('Nearest-height selection requires integer indices')
        return outputs.gather(2, h.long()[:, None, None, None].expand(-1, outputs.shape[1], 1, 2)).squeeze(2)
    if mode != 'linear':
        raise ValueError('Unknown height interpolation mode')
    low = torch.floor(h).long()
    high = (low + 1).clamp(max=6)
    lo = outputs.gather(2, low[:, None, None, None].expand(-1, outputs.shape[1], 1, 2)).squeeze(2)
    hi = outputs.gather(2, high[:, None, None, None].expand(-1, outputs.shape[1], 1, 2)).squeeze(2)
    w = (h - low).to(outputs.dtype)[:, None, None]
    return lo + (hi - lo) * w
