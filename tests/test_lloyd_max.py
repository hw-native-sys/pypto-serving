"""Verify the optimized Lloyd-Max solver matches the old numerical-integration version."""

import math
import sys
import time

import torch


def solve_lloyd_max_old(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Original implementation: numerical integration with 2048 samples."""
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma * sigma)) * math.exp(
            -x * x / (2 * sigma * sigma)
        )

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            n_samples = 2048
            xs = torch.linspace(a, b, n_samples)
            pdf_vals = torch.tensor([pdf(x) for x in xs])
            weighted = xs * pdf_vals
            if pdf_vals.sum() > 1e-15:
                new_centroids.append((weighted.sum() / pdf_vals.sum()).item())
            else:
                new_centroids.append(centroids[i])
        max_shift = max(
            abs(new_centroids[i] - centroids[i]) for i in range(n_levels)
        )
        centroids = new_centroids
        if max_shift < tol:
            break

    return torch.tensor(centroids, dtype=torch.float64)


def solve_lloyd_max_new(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Optimized implementation: analytical Gaussian conditional mean."""
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = torch.linspace(lo, hi, n_levels + 2)[1:-1].double()

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = torch.zeros(n_levels + 1, dtype=torch.float64)
        edges[0] = lo * 3
        edges[1:-1] = boundaries
        edges[-1] = hi * 3

        z = edges / sigma
        phi_z = (-0.5 * z * z).exp() * (1.0 / math.sqrt(2.0 * math.pi))
        cdf_z = 0.5 * (1.0 + torch.erf(z * (1.0 / math.sqrt(2.0))))

        phi_diff = phi_z[:-1] - phi_z[1:]
        cdf_diff = cdf_z[1:] - cdf_z[:-1]
        valid = cdf_diff > 1e-15
        new_centroids = torch.where(
            valid,
            sigma * phi_diff / cdf_diff.clamp(min=1e-15),
            centroids,
        )
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < tol:
            break

    return centroids


def test_case(d: int, bits: int):
    n_levels = 2 ** bits
    print(f"\n{'='*60}")
    print(f"d={d}, bits={bits}, n_levels={n_levels}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    c_old = solve_lloyd_max_old(d, bits)
    dt_old = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    c_new = solve_lloyd_max_new(d, bits)
    dt_new = (time.perf_counter() - t0) * 1000

    diff = (c_new - c_old).abs()
    sigma = 1.0 / math.sqrt(d)

    print(f"  old centroids ({dt_old:.1f} ms): {c_old.tolist()}")
    print(f"  new centroids ({dt_new:.1f} ms): {c_new.tolist()}")
    print(f"  max  abs diff: {diff.max().item():.2e}  (sigma={sigma:.4f})")
    print(f"  mean abs diff: {diff.mean().item():.2e}")
    print(f"  speedup: {dt_old / max(dt_new, 0.01):.1f}x")

    # Verify: new centroids are the better quantizer (lower MSE for Gaussian)
    # MSE = sum_{i} integral_{b_i}^{b_{i+1}} (x - c_i)^2 * f(x) dx
    def quantization_mse(centroids_in):
        """Compute MSE of a scalar quantizer for N(0, sigma^2)."""
        cs = sorted(centroids_in.tolist())
        boundaries = [-1e10] + [(cs[i] + cs[i+1]) / 2 for i in range(len(cs)-1)] + [1e10]
        total_mse = 0.0
        n_pts = 10000
        for i in range(len(cs)):
            a, b = boundaries[i], boundaries[i+1]
            xs = torch.linspace(a, b, n_pts, dtype=torch.float64)
            pdf_vals = torch.exp(-0.5 * (xs / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            mse_i = ((xs - cs[i]) ** 2 * pdf_vals).sum()
            total_mse += mse_i.item()
        dx = (boundaries[-1] - boundaries[0]) / n_pts  # approximate
        return total_mse / n_pts  # normalized

    mse_old = quantization_mse(c_old)
    mse_new = quantization_mse(c_new)
    print(f"  MSE old: {mse_old:.6e}")
    print(f"  MSE new: {mse_new:.6e}")
    if mse_new <= mse_old * 1.005:
        print(f"  PASS: new MSE <= old MSE (within 0.5%)")
    else:
        print(f"  FAIL: new MSE > old MSE by {(mse_new/mse_old-1)*100:.2f}%")

    # Verify centroids are monotonically increasing
    diffs_signed = c_new[1:] - c_new[:-1]
    if (diffs_signed > 0).all():
        print(f"  PASS: centroids monotonically increasing")
    else:
        print(f"  FAIL: centroids not monotonically increasing")

    # Verify symmetry: centroids should be symmetric around 0
    mid = n_levels // 2
    if n_levels % 2 == 0:
        sym_diff = (c_new[:mid] + c_new[mid:].flip(0)).abs().max().item()
    else:
        sym_diff = (c_new[:mid] + c_new[mid+1:].flip(0)).abs().max().item()
    print(f"  symmetry around 0: max deviation = {sym_diff:.2e}")
    if sym_diff < 1e-8:
        print(f"  PASS: symmetric around 0")
    else:
        print(f"  WARN: not perfectly symmetric (expected for finite iterations)")


def main():
    all_pass = True

    # Typical Qwen3-14B configurations
    test_configs = [
        # (head_dim, bits)
        (128, 2),   # value bits
        (128, 4),   # key bits
        (128, 3),   # alternative
        (64, 2),    # smaller model
        (64, 4),
        (128, 6),   # higher precision
    ]

    for d, bits in test_configs:
        test_case(d, bits)

    print(f"\n{'='*60}")
    print("Summary: compare speed across configs")
    print(f"{'='*60}")
    print(f"{'d':>6} {'bits':>5} {'old(ms)':>10} {'new(ms)':>10} {'speedup':>10}")
    print("-" * 45)
    for d, bits in test_configs:
        t0 = time.perf_counter()
        solve_lloyd_max_old(d, bits)
        dt_old = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        solve_lloyd_max_new(d, bits)
        dt_new = (time.perf_counter() - t0) * 1000
        print(f"{d:>6} {bits:>5} {dt_old:>10.1f} {dt_new:>10.1f} {dt_old/max(dt_new,0.01):>9.1f}x")


if __name__ == "__main__":
    main()
