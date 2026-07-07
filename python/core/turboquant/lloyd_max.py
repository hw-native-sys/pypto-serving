"""Lloyd-Max optimal scalar quantizer for the Gaussian distribution.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate follows approximately N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.
"""

import math

import torch


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d).

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    # Gaussian PDF: N(0, sigma^2)
    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma * sigma)) * math.exp(
            -x * x / (2 * sigma * sigma)
        )

    for _ in range(max_iter):
        # Step 1: Compute boundaries (midpoints)
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]

        # Step 2: Update centroids as conditional expectations
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            # Numerical integration via sampling (avoids scipy dependency)
            n_samples = 2048
            xs = torch.linspace(a, b, n_samples)
            pdf_vals = torch.tensor([pdf(x) for x in xs])
            weighted = xs * pdf_vals
            if pdf_vals.sum() > 1e-15:
                new_centroids.append((weighted.sum() / pdf_vals.sum()).item())
            else:
                new_centroids.append(centroids[i])

        # Check convergence
        max_shift = max(
            abs(new_centroids[i] - centroids[i]) for i in range(n_levels)
        )
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given dimension and bit-width."""

    def __init__(self, d: int, bits: int):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits)

    def __repr__(self):
        return f"LloydMaxCodebook(d={self.d}, bits={self.bits}, levels={self.n_levels})"
