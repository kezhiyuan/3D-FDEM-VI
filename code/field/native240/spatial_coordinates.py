import math
import numpy as np
import torch

class CosineSpatialCoordinates:

    def __init__(self, shape_yxz, lengths_yxz=(8.0, 8.0, 3.0), epsilon=0.125, *, device='cpu', dtype=torch.float32):
        self.shape = tuple((int(n) for n in shape_yxz))
        self.lengths = tuple((float(v) for v in lengths_yxz))
        self.epsilon = float(epsilon)
        if len(self.shape) != 3 or len(self.lengths) != 3 or (not 0 < self.epsilon <= 1) or (min(self.lengths) < 0):
            raise ValueError('Positive, invertible three-axis coordinates required')
        self.identity = all((v == 0 for v in self.lengths))
        self.bases = []
        exponents = []
        for (n, length) in zip(self.shape, self.lengths):
            k = np.arange(n, dtype=np.float64)
            x = np.arange(n, dtype=np.float64)
            basis = np.cos(math.pi * k[:, None] * (x[None, :] + 0.5) / n) * math.sqrt(2.0 / n)
            basis[0] /= math.sqrt(2.0)
            self.bases.append(torch.as_tensor(basis, device=device, dtype=dtype))
            exponents.append(torch.as_tensor((math.pi * k * length / n) ** 2, device=device, dtype=dtype))
        decay = torch.exp(-0.5 * (exponents[0][:, None, None] + exponents[1][None, :, None] + exponents[2][None, None, :]))
        self.weight = self.epsilon + (1 - self.epsilon) * decay

    def _transform(self, a, inverse=False):
        if tuple(a.shape[-3:]) != self.shape:
            raise ValueError('Wrong native coordinate shape')
        for (axis, basis) in zip(range(a.ndim - 3, a.ndim), self.bases):
            b = a.movedim(axis, -1)
            a = (b @ (basis if inverse else basis.T)).movedim(-1, axis)
        return a

    def apply(self, z):
        if self.identity:
            return z
        return self._transform(self._transform(z) * self.weight, inverse=True)

    def inverse(self, logits):
        if self.identity:
            return logits
        return self._transform(self._transform(logits) / self.weight, inverse=True)

    def rebase(self, z, next_coordinates):
        return next_coordinates.inverse(self.apply(z))

    def metadata(self):
        return {'definition': 'T=C^T diag(epsilon+(1-epsilon)exp(-0.5 sum((pi*k*l/N)^2))) C', 'boundary': 'orthonormal DCT-II / Neumann cosine basis', 'shape_yxz': list(self.shape), 'lengths_native_cell_units_yxz': list(self.lengths), 'epsilon': self.epsilon, 'minimum_spectral_weight': float(self.weight.min()), 'maximum_spectral_weight': float(self.weight.max()), 'all_native_degrees_free': True, 'changes_posterior_target': False, 'vertical_lengths_are_optimization_coordinates_not_physical_prior_lengths': True}
