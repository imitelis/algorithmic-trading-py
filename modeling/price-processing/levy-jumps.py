import numpy as np

def simulate_levy_jumps(T=1.0, lambda_jump=5, mu_jump=0, sigma_jump=1, dt=0.01):
    """
    Simulate a Lévy process as a compound Poisson process with normal jumps.

    Parameters:
    - T: Total time
    - lambda_jump: Jump intensity (mean number of jumps per unit time)
    - mu_jump: Mean of jump size
    - sigma_jump: Std dev of jump size
    - dt: Time step size

    Returns:
    - times: Array of time steps
    - X: Simulated Lévy process path
    """
    N = int(T / dt)
    times = np.linspace(0, T, N)
    X = np.zeros(N)

    for i in range(1, N):
        # Check for a jump in this small interval
        num_jumps = np.random.poisson(lambda_jump * dt)
        if num_jumps > 0:
            # Add jump(s)
            jump_sizes = np.random.normal(mu_jump, sigma_jump, size=num_jumps)
            X[i] = X[i-1] + np.sum(jump_sizes)
        else:
            X[i] = X[i-1]

    return times, X