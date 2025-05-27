import numpy as np

def simulate_random_walk(S0, mu=0, sigma=1, N=252, n_paths=5):
    """
    Simulate a simple random walk (discrete-time) for asset prices.

    Parameters:
    - S0: Initial asset price
    - mu: Drift term (average step size)
    - sigma: Volatility (standard deviation of step size)
    - N: Number of time steps
    - n_paths: Number of independent paths to simulate

    Returns:
    - prices: Simulated price paths as a (N+1 x n_paths) array
    """
    
    dt = 1
    steps = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=(N, n_paths))
    prices = S0 + np.cumsum(steps, axis=0)
    prices = np.vstack([np.full(n_paths, S0), prices])  # Add initial price
    return prices