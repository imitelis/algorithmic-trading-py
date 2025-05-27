import numpy as np

def simulate_ou_process(x0, theta, mu, sigma, T=1, N=252, n_paths=5):
    """
    Simulate paths of an Ornstein-Uhlenbeck (OU) process.

    Parameters:
    - x0: Initial value (starting point of the process)
    - theta: Rate of mean reversion (speed of reversion to mu)
    - mu: Long-term mean (the level to which the process reverts)
    - sigma: Volatility (standard deviation of the process)
    - T: Total time (in years)
    - N: Number of time steps
    - n_paths: Number of simulated paths

    Returns:
    - X: Simulated process paths (N+1 x n_paths)
    """
    
    dt = T / N
    X = np.zeros((N + 1, n_paths))
    X[0] = x0

    for t in range(1, N + 1):
        Z = np.random.normal(size=n_paths)  # Standard normal random shocks
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) * dt + sigma * np.sqrt(dt) * Z

    return X
