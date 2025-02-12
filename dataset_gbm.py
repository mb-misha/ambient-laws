import numpy as np
import torch
from torch.utils.data import Dataset


class GBMGenerativeDataset(Dataset):
    def __init__(
            self,
            n_paths=10000,
            n_steps=200,
            n_ts_features=1,
            s_price=100.0,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            return_log_returns=False,
    ):
        """
        Generates GBM paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        self.n_ts_features = n_ts_features
        self.return_log_returns = return_log_returns
        self.n_paths = n_paths
        self.n_steps = n_steps
        dt = T / n_steps

        # Simulate GBM paths
        paths = self.simulate_gbm_paths(n_paths, n_steps-1, s_price, mu, sigma, dt)
        self.paths = paths.reshape(n_paths, n_steps, n_ts_features)

        if return_log_returns:
            self.paths = self.compute_log_returns()



        self.dt = dt  # Store dt for parameter estimation
        self.name = 'GBMGenerativeDataset'
        self.resolution = n_steps

    def simulate_gbm_paths(self, n_paths, n_steps, S0, mu, sigma, dt):
        """
        Simulates Geometric Brownian Motion (GBM) paths.
        """
        paths = np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(n_paths, n_steps))
        )
        # Add ones to the beginning
        paths = np.hstack([np.ones((n_paths, 1)), paths])
        paths = S0 * paths.cumprod(axis=1)  # Compute cumulative product to get paths
        return paths

    def compute_log_returns(self):
        """
        Computes the log returns of the paths.
        """
        log_returns = np.diff(np.log(self.paths), axis=1)
        return log_returns


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.tensor(self.paths[idx], dtype=torch.float32)



def estimate_parameters(paths, dt):
    """
    Estimates the drift (μ) and volatility (σ) from the generated paths.
    """
    paths = np.log(paths)
    sigma = np.sqrt(
        (np.diff(paths, axis=1) ** 2).sum(axis=1) / (paths.shape[1] * dt)
    )
    log_mu = (paths[:, -1] - paths[:, 0]) / (paths.shape[1] * dt)
    mu = log_mu + 0.5 * sigma ** 2
    return mu, sigma


def reverse_log_return(log_returns, s0):
    n_paths = log_returns.shape[0]
    n_steps = log_returns.shape[1] + 1
    reconstructed_prices = np.zeros((n_paths, n_steps))
    reconstructed_prices[:, 0] = s0
    for i in range(1, n_steps):
        reconstructed_prices[:, i] = reconstructed_prices[:, i - 1] * np.exp(log_returns[:, i - 1])
    return reconstructed_prices

