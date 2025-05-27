import numpy as np

def simulate_gbm(S0, mu, sigma, T=1, N=252, n_paths=10):
    dt = T / N
    prices = np.zeros((N, n_paths))
    prices[0] = S0
    
    for t in range(1, N):
        Z = np.random.standard_normal(n_paths)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
    return prices
