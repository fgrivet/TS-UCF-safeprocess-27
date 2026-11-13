"""Calculation of window size and distances, UCF score,
calculation of the impact score and the optimal polynomial.
"""

import logging
import math
from collections import Counter
from functools import reduce

import numpy as np
from scipy.fft import fft

logger = logging.getLogger("TS_UCF.core")


def compute_l_star(numbers):
    def gcd_of_two(a, b):
        gcd = math.gcd(a, b)
        if gcd == 1:
            return a * b
        else:
            return gcd

    return reduce(gcd_of_two, numbers)


def infer_window_size(X: np.ndarray, min_windows=10, min_window_size=10) -> int:
    N = len(X)
    Xf = fft(X)

    magnitude = np.abs(Xf[: N // 2])  # type: ignore
    magnitude[:min_windows] = 0  # Set the average and the first magnitudes to 0 to get at least min_windows windows
    magnitude[N // min_window_size :] = 0  # Set the average and the first magnitudes to 0 to get at least min_windows windows

    # 1 magnitude per channel
    max_per_channel = magnitude.argmax(axis=0)
    logger.debug("magnitude max = %s", max_per_channel)

    # Try to find a common value
    l = compute_l_star(max_per_channel)
    logger.debug("l found = %d", l)

    if l < min_windows or l > (N // min_windows):
        c = Counter(max_per_channel)
        l = max(c, key=c.get)  # type: ignore
        logger.debug("l max = %d", l)

    L = int(np.floor(N / l))

    logger.info("Window size infered = %d", L)

    return L


def infer_window_size_acf(X: np.ndarray, window_sizes: list[int]) -> int:
    res = []
    N = len(X)
    for ws in window_sizes:
        if ws > N // 4:  # At least 4 windows
            res.append(np.inf)
        else:
            w = make_windows(X, ws, ws)[:, :, 0]
            distances = pairwise_dist_rfft(w, w).mean()
            res.append(distances)

    l = np.argmin(res)
    L = window_sizes[l]
    return L


def make_windows(X: np.ndarray, L: int, s: int | None = None) -> np.ndarray:
    if len(X) < L:
        raise ValueError(f"Signal is too short (length={len(X)}) to create windows of length {L}.")
    if s is None:
        s = L
    elif s > L:
        raise ValueError(f"Shift ({s}) is larger than the window size ({L}).")
    return np.lib.stride_tricks.sliding_window_view(X, L, axis=0)[::s].swapaxes(1, 2)  # (n_windows, window_size=L, n_channel=k)


def pairwise_sqeuclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_sq = np.einsum("ij,ij->i", A, A)[:, None]
    B_sq = np.einsum("ij,ij->i", B, B)[None, :]

    D = A @ B.T
    D *= -2.0
    D += A_sq
    D += B_sq
    np.maximum(D, 0, out=D)  # numerical safety (D must be >= 0)
    return D


def pairwise_dist_rfft(A, B):
    """
    Calculates the matrix of squared pairwise distances between the MODULES of the Fourier spectra (rfft).
    The same frequency with a phase shift (e.g., cosine vs. sine) will result in a distance of 0.

    A_temporal: shape (K, N) - Real signals
    B_temporal: shape (K', N) - Real signals
    Returns: shape (K, K')
    """
    N = A.shape[1]

    # 1. Calculating the MODULE of the rfft (Returns REAL and positive values)
    # Resulting shapes: (K, N//2 + 1) and (K', N//2 + 1)
    A_mod = np.abs(np.fft.rfft(A, axis=1))
    B_mod = np.abs(np.fft.rfft(B, axis=1))

    # 2. Weighting vector for spectral symmetry (rfft)
    # Intermediate frequencies count double; the endpoints (DC and Nyquist) count once
    poids = np.full(A_mod.shape[1], 2.0)
    poids[0] = 1.0
    if N % 2 == 0:
        poids[-1] = 1.0

    # 3. Applying the weights
    # Since we are working with (real) modules, we apply the ‘weights’ directly when calculating
    # the norms and the cross-product.

    # Squared norms of each row (weighted) -> forms (K,) and (K',)
    norm_A = np.sum((A_mod**2) * poids, axis=1)
    norm_B = np.sum((B_mod**2) * poids, axis=1)

    # Weighted cross-product -> shape (K, K')
    # Multiply the columns of B_mod by ‘weights’ before performing the matrix multiplication
    produit_croise = np.dot(A_mod, (B_mod * poids).T)

    # 4. Final combination using NumPy broadcasting
    dist_matrix = norm_A[:, None] + norm_B[None, :] - 2 * produit_croise

    # Division by N (FFT normalization) + digital safety for values close to 0
    return np.maximum(dist_matrix, 0) / N


def ucf_score(distances: np.ndarray, n_min: int, n_max: int, breakdown_tol: float = 0.0) -> np.ndarray:
    distances = np.ascontiguousarray(distances)
    if distances.ndim != 2:
        raise ValueError("D must have shape (B, N)")

    B, N = distances.shape
    dtype = distances.dtype

    if n_min < 0 or n_max < n_min:
        raise ValueError("Need 0 <= n_min <= n_max")
    if n_max >= N:
        raise ValueError("Need n_max < N; more precisely, degree n requires at least n+1 distinct support points")

    q_prev = np.zeros((B, N), dtype=dtype)
    q = np.ones((B, N), dtype=dtype)

    q_norm = np.linalg.norm(q, axis=1)
    if np.any(~np.isfinite(q_norm)) or np.any(q_norm == 0):
        raise FloatingPointError("Initial normalization failed.")

    q /= q_norm[:, None]

    beta_prev = np.zeros(B, dtype=dtype)
    p_prev = np.zeros(B, dtype=dtype)
    p = np.ones(B, dtype=dtype)

    score = np.ones(B, dtype=dtype)
    scores_all = np.full((B, n_max + 1), np.nan, dtype=dtype)
    scores_all[:, 0] = score

    active = np.ones(B, dtype=bool)

    for j in range(n_max):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        q_i = q[idx]
        q_prev_i = q_prev[idx]
        d_i = distances[idx]
        beta_prev_i = beta_prev[idx]

        # One step of the Lanczos three-term recurrence: orthogonalize D @ q_i against the
        # current and previous Lanczos vectors to build the (unnormalized) next one.
        u = d_i * q_i
        w = u - beta_prev_i[:, None] * q_prev_i

        alpha = np.einsum("bn,bn->b", q_i, w)

        z = w - alpha[:, None] * q_i
        beta = np.linalg.norm(z, axis=1)

        # A window "breaks down" when its residual can no longer be normalized into a new
        # Lanczos vector; freeze it (its scores for degrees j+1..n_max stay NaN).
        bad = ~np.isfinite(beta) | (beta <= breakdown_tol)
        if np.any(bad):
            active[idx[bad]] = False

        good_local = ~bad
        if not np.any(good_local):
            continue

        gidx = idx[good_local]

        beta_g = beta[good_local]
        alpha_g = alpha[good_local]

        # p tracks the numerator of the continued-fraction recursion for the (0, 0) entry of
        # the inverse tridiagonal (Jacobi) matrix; `score` accumulates sum_i p_i^2, which is
        # exactly Lambda_n(0)^{-1} at degree n = j + 1.
        p_next = (-alpha_g * p[gidx] - beta_prev[gidx] * p_prev[gidx]) / beta_g

        score[gidx] = score[gidx] + p_next * p_next
        scores_all[gidx, j + 1] = score[gidx]

        q_next = z[good_local] / beta_g[:, None]

        q_prev[gidx] = q[gidx]
        q[gidx] = q_next
        beta_prev[gidx] = beta_g

        p_prev[gidx] = p[gidx]
        p[gidx] = p_next

    result = scores_all[:, n_min : n_max + 1]

    n_affected = int(np.isnan(result).any(axis=1).sum())
    if n_affected > 0:
        logger.warning(
            "Lanczos breakdown detected for %d/%d window(s) in degree range [%d, %d]; "
            "their scores are NaN from the breakdown degree onward. "
            "This happens when a window's N normalized distances do not support orthogonal "
            "polynomials up to degree n_max (e.g. many duplicate/near-duplicate values). "
            "Consider lowering n_max for this scale.",
            n_affected,
            B,
            n_min,
            n_max,
        )

    return result


def impact_scores(distances, sigma: float):
    K_train = distances.shape[2]

    # Number of distances in the first sigma quantile (at least one distance)
    n_dist_kappa = max(1, int(sigma * K_train))

    # For each window test select the n_dist_kappa smallest multivariate (-1) distances.
    # np.argpartition (O(N)) is enough since we only need the smallest, not a full ranking.
    kappa = np.argpartition(distances[-1], n_dist_kappa - 1, axis=1)[:, :n_dist_kappa]  # (K_test, n_dist_kappa)

    # Distances of each parameter at the indices in kappa
    d_kappa_param = np.take_along_axis(distances[:-1], kappa.reshape(1, -1, n_dist_kappa))  # (k, K_test, n_dist_kappa)

    # Multivariate distances at the indices in kappa
    d_kappa_multi = np.take_along_axis(distances[-1], kappa)  # (K_test, n_dist_kappa)

    idx_0_dist_multi = np.nonzero(d_kappa_multi == 0)
    div = d_kappa_param / d_kappa_multi
    div[:, idx_0_dist_multi] = 0  # If a test window was in the reference database, then its multivariate distance = 0, so we replace the nan by 0

    # Median contribution for each parameter for each window, transpose to get the correct dimensions
    contributions = np.median(div, axis=2).T  # (K_test, k)

    # Normalize contributions to proportions summing to 1.
    total = contributions.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0  # avoid 0/0 when every member's contributions is exactly 0
    pct = contributions / total  # (K_test, k)

    return pct


def compute_optimal_polynomial(distances: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    powers = np.arange(0, n + 1).reshape(1, 1, n + 1)
    V = np.pow(distances[:, :, None], powers)
    G = np.swapaxes(V, 1, 2) @ V
    G /= distances.shape[1]

    G = (G + np.swapaxes(G, 1, 2)) / 2  # Ensure symmetry

    v = np.zeros(n + 1)
    v[0] = 1
    v = np.broadcast_to(v, (len(distances), n + 1)).reshape((len(distances), n + 1, 1))

    def solve_3D(a, b):
        return np.stack([np.linalg.solve(a[i], b[i]) for i in range(len(a))], axis=0)

    num = solve_3D(G + np.eye(n + 1) * 1e-10, v)[:, :, 0]

    denom = np.einsum("i,mi->m", v[0, :, 0], num).reshape(-1, 1)

    correct_scores = ucf_score(distances, n, n)
    errors_str = [f"Window {i+1}: {abs(denom[i] - correct_scores[i])[0]:.6g}" for i in range(len(denom))]
    logger.info("Impact of the inversion error on the score for each window: %s", errors_str)

    p_opt = num / denom

    return p_opt, correct_scores, denom
