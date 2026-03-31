from __future__ import annotations

import math

import numpy as np
from scipy.special import betaln, logsumexp
from scipy.stats import qmc


def _latent_samples(sample_count: int, model_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = qmc.Sobol(d=2 * model_count, scramble=True, seed=seed)
    power = math.ceil(math.log2(max(2, sample_count)))
    uniforms = sampler.random_base2(power)[:sample_count]
    uniforms = np.clip(uniforms, 1e-6, 1.0 - 1e-6)
    return uniforms[:, :model_count], uniforms[:, model_count:]


def _mu_from_b(b_samples: np.ndarray, truth_value: int) -> np.ndarray:
    mu = truth_value + (b_samples - truth_value) * np.abs(2.0 * b_samples - 1.0)
    return np.clip(mu, 1e-6, 1.0 - 1e-6)


def _log_g_matrix(
    s_matrix: np.ndarray,
    n_matrix: np.ndarray,
    b_samples: np.ndarray,
    tau_samples: np.ndarray,
    truth_value: int,
) -> np.ndarray:
    sample_count, model_count = b_samples.shape
    log_g = np.zeros((sample_count, s_matrix.shape[0]), dtype=np.float64)
    mu = _mu_from_b(b_samples, truth_value)
    nu = 1.0 / (1.0 - np.clip(tau_samples, 1e-6, 1.0 - 1e-6))

    for model_index in range(model_count):
        mu_nu = (mu[:, model_index] * nu[:, model_index])[:, None]
        one_minus_mu_nu = ((1.0 - mu[:, model_index]) * nu[:, model_index])[:, None]
        s_values = s_matrix[:, model_index][None, :]
        n_values = n_matrix[:, model_index][None, :]
        log_g += betaln(s_values + mu_nu, n_values - s_values + one_minus_mu_nu) - betaln(mu_nu, one_minus_mu_nu)

    return log_g


def posterior_density(
    s_matrix: np.ndarray,
    n_matrix: np.ndarray,
    p_grid: np.ndarray,
    sample_count: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    if s_matrix.shape != n_matrix.shape:
        raise ValueError("S and N matrices must have the same shape.")
    if s_matrix.shape[0] == 0:
        raise ValueError("Cannot compute a posterior with zero events.")

    b_samples, tau_samples = _latent_samples(sample_count=sample_count, model_count=s_matrix.shape[1], seed=seed)
    log_g_yes = _log_g_matrix(s_matrix, n_matrix, b_samples, tau_samples, truth_value=1)
    log_g_no = _log_g_matrix(s_matrix, n_matrix, b_samples, tau_samples, truth_value=0)

    log_density = np.empty_like(p_grid)
    for start in range(0, p_grid.size, chunk_size):
        stop = min(start + chunk_size, p_grid.size)
        p_chunk = p_grid[start:stop]
        log_p = np.log(np.clip(p_chunk, 1e-12, 1.0))
        log_one_minus_p = np.log1p(-np.clip(p_chunk, 0.0, 1.0 - 1e-12))
        mixed = np.logaddexp(
            log_g_yes[:, :, None] + log_p[None, None, :],
            log_g_no[:, :, None] + log_one_minus_p[None, None, :],
        )
        log_density[start:stop] = logsumexp(mixed.sum(axis=1), axis=0) - math.log(sample_count)

    log_density -= np.max(log_density)
    density = np.exp(log_density)
    density /= np.trapezoid(density, p_grid)
    return density
