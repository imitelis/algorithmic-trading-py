import numpy as np

def simulate_vasicek(r0, a, b, sigma, T=1, N=252, n_paths=5):
    """
    Simulate short rate paths using the Vasicek mean-reverting model.

    Parameters:
    - r0: Initial interest rate
    - a: Speed of mean reversion
    - b: Long-term mean level
    - sigma: Volatility of the short rate
    - T: Total time (in years)
    - N: Number of time steps
    - n_paths: Number of simulated paths

    Returns:
    - rates: Simulated interest rate paths (N+1 x n_paths)
    """

    dt = T / N
    rates = np.zeros((N + 1, n_paths))
    rates[0] = r0

    for t in range(1, N + 1):
        Z = np.random.normal(size=n_paths)  # Standard normal shocks
        r_prev = rates[t - 1]

        # Vasicek model dynamics: dr = a(b - r)dt + sigma*dW
        rates[t] = r_prev + a * (b - r_prev) * dt + sigma * np.sqrt(dt) * Z

    return rates
