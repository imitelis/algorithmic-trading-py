import numpy as np

def simulate_cir_process(r0, kappa, theta, sigma, T=1, N=252, n_paths=5):
    """
    Simulate paths of a Cox-Ingersoll-Ross (CIR) process.

    Parameters:
    - r0: Initial value of the process (starting point, e.g., initial interest rate)
    - kappa: Speed of mean reversion (how fast the process reverts to the mean)
    - theta: Long-term mean (the level to which the process reverts)
    - sigma: Volatility (volatility of the process)
    - T: Total time (in years)
    - N: Number of time steps
    - n_paths: Number of simulated paths

    Returns:
    - r: Simulated process paths (N+1 x n_paths)
    """
    dt = T / N
    r = np.zeros((N + 1, n_paths))
    r[0] = r0

    for t in range(1, N + 1):
        Z = np.random.normal(size=n_paths)  # Standard normal random shocks
        r_prev = np.maximum(r[t - 1], 0)  # Ensure non-negative rates

        # CIR process dynamics: dr = kappa(theta - r) dt + sigma * sqrt(r) * dW
        r[t] = r_prev + kappa * (theta - r_prev) * dt + sigma * np.sqrt(r_prev) * np.sqrt(dt) * Z

    return r
