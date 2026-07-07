"""Lloyd-Max optimal scalar quantizer for the Gaussian distribution.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate follows approximately N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.
"""

import contextlib
import math

import torch

# Lloyd-Max codebooks are a deterministic function of (d, bits); many layers
# share the same bit-width, so cache the solved codebook instead of recomputing
# it (the per-layer compressors create one codebook each — 80+ for Qwen3-14B).
_CODEBOOK_CACHE: dict[tuple[int, int], "LloydMaxCodebook"] = {}


@contextlib.contextmanager
def limited_cpu_threads(n: int = 8):
    """Temporarily cap torch's CPU thread count.

    On hosts with many cores torch defaults to using all of them, which causes
    severe thread-contention overhead on the small CPU tensors used during
    TurboQuant setup (QR decomposition, Lloyd-Max numerical integration) —
    observed >100x slowdown at 320 threads vs 8. The inference workload itself
    runs on the NPU and does not depend on torch CPU threading, so capping
    threads during these setup-only computations is safe.
    """
    prev = torch.get_num_threads()
    if prev > n:
        torch.set_num_threads(n)
    try:
        yield
    finally:
        if prev != torch.get_num_threads():
            torch.set_num_threads(prev)


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d).

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    var = sigma * sigma
    # Precompute the Gaussian N(0, sigma^2) normalization once.
    norm = 1.0 / math.sqrt(2.0 * math.pi * var)

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma].
    lo, hi = -3.5 * sigma, 3.5 * sigma
    idx = torch.arange(n_levels, dtype=torch.float64)
    centroids = lo + (hi - lo) * (idx + 0.5) / n_levels

    # Integration grid shared across all levels each iteration. Using float64
    # keeps the conditional-expectation integration accurate; the result is cast
    # back to float32 at the end.
    n_samples = 2048
    grid = torch.linspace(0.0, 1.0, n_samples, dtype=torch.float64)

    with limited_cpu_threads():
        for _ in range(max_iter):
            # Step 1: boundaries are midpoints between adjacent centroids.
            boundaries = (centroids[:-1] + centroids[1:]) / 2.0
            edges = torch.cat(
                [
                    torch.tensor([lo * 3.0], dtype=torch.float64),
                    boundaries,
                    torch.tensor([hi * 3.0], dtype=torch.float64),
                ]
            )
            a = edges[:-1]  # (n_levels,)
            b = edges[1:]  # (n_levels,)

            # Step 2: update each centroid as the conditional expectation over
            # its bucket [a, b]. Build the (n_levels, n_samples) sample grid in
            # one shot and evaluate the Gaussian density vectorized (the
            # pure-Python loop over samples/levels here previously dominated
            # KVCompressor construction).
            xs = a.unsqueeze(1) + (b - a).unsqueeze(1) * grid  # (n_levels, n_samples)
            pdf_vals = norm * torch.exp(-xs * xs / (2.0 * var))  # (n_levels, n_samples)
            weight = pdf_vals.sum(dim=1)
            new_centroids = (xs * pdf_vals).sum(dim=1) / weight.clamp_min(1e-15)
            # Buckets with negligible probability keep their previous centroid.
            new_centroids = torch.where(weight > 1e-15, new_centroids, centroids)

            max_shift = (new_centroids - centroids).abs().max().item()
            centroids = new_centroids
            if max_shift < tol:
                break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return (
        centroids.to(torch.float32),
        boundaries.to(torch.float32),
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


def get_codebook(d: int, bits: int) -> LloydMaxCodebook:
    """Return a cached LloydMaxCodebook for (d, bits), solving it once."""
    key = (d, bits)
    cb = _CODEBOOK_CACHE.get(key)
    if cb is None:
        cb = LloydMaxCodebook(d, bits)
        _CODEBOOK_CACHE[key] = cb
    return cb
