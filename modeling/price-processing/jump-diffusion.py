import numpy as np

def simulate_jump_diffusion(S0, mu, sigma, lamb, m, delta, T=1, N=252, n_paths=5):
    """
    Simulate asset paths using a Merton jump-diffusion model.

    Parameters:
    - S0: Initial asset price
    - mu: Expected return (drift)
    - sigma: Volatility of the diffusion (Brownian motion)
    - lamb: Jump intensity (expected number of jumps per unit time)
    - m: Mean of log-normal jump size
    - delta: Std dev of log-normal jump size
    - T: Total time (in years)
    - N: Number of time steps
    - n_paths: Number of simulated paths

    Returns:
    - prices: Simulated price paths as a (N x n_paths) array
    """
    dt = T / N
    prices = np.zeros((N, n_paths))
    prices[0] = S0

    for t in range(1, N):
        Z = np.random.standard_normal(n_paths)        # Brownian shocks
        J = np.random.normal(m, delta, n_paths)       # Jump sizes
        N_jumps = np.random.poisson(lamb * dt, n_paths)  # Poisson jumps

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        jumps = (np.exp(J) - 1) * N_jumps

        prices[t] = prices[t-1] * np.exp(drift + diffusion + jumps)

    return prices
