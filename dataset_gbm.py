import numpy as np
import torch
from torch.utils.data import Dataset


class StochasticModelDataset(Dataset):
    def __init__(
        self,
        n_paths=10000,
        n_steps=200,
        n_ts_features=1,
        T=1.0,
        return_log_returns=False,
        normalize=None,
    ):
        """
        Base class for stochastic model path generators with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        self.n_ts_features = n_ts_features
        self.return_log_returns = return_log_returns
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.normalize = normalize
        self.dt = T/n_steps
        assert n_ts_features == 1, "Only 1D timeseries data is supported."
        assert self.normalize in [None, 'global_zscore', 'per_path_zscore', 'global_mean']
        
        self.paths = None
        
        # unet compatibility
        self.resolution = self.n_steps
        self.num_channels = 1
        self.label_dim = 0
        self.has_labels = False
        self.has_onehot_labels = False
    
    def simulate_paths(self):
        """
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def compute_log_returns(self):
        """
        Computes the log returns of the paths.
        """
        log_returns = np.diff(np.log(self.paths), axis=-1)
        return log_returns

    def process_paths(self):
        """
        Process the generated paths and handle log returns if needed.
        """
        if self.return_log_returns:
            self.paths = self.compute_log_returns()
            self.n_steps -= 1
        self.paths = np.expand_dims(self.paths, axis=1)
        assert self.paths.shape == (self.n_paths, 1, self.n_steps)
        self.resolution = self.n_steps
        self.mean = np.mean(self.paths)
        self.std = np.std(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ts = self.paths[idx]
        if self.normalize == 'global_zscore':
            ts = (ts - self.mean) / self.std
        elif self.normalize == 'per_path_zscore':
            ts = (ts - np.mean(ts, axis=-1, keepdims=True)) / np.std(ts, axis=-1, keepdims=True)
        elif self.normalize == 'global_mean':
            ts = ts/self.mean
        return {
            "image": ts.copy(),
            "label": np.zeros(0, dtype=np.float32),
            'sigma': 0.0,
            'noise': np.zeros_like(self.paths[idx], dtype=np.float32),
            'corruption_mask': np.zeros_like(self.paths[idx], dtype=np.float32),
        }

class GBMGenerativeDataset(StochasticModelDataset):
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
            normalize=None,
    ):
        """
        Generates GBM paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize)
        self.s_price = s_price
        self.mu = mu
        self.sigma = sigma
        self.name = 'GBMGenerativeDataset'
        
        # Simulate GBM paths
        self.paths = self.simulate_paths()
        self.process_paths()

    def simulate_paths(self):
        """
        Simulates Geometric Brownian Motion (GBM) paths.
        """
        return self.simulate_gbm_paths(self.n_paths, self.n_steps-1, self.s_price, self.mu, self.sigma, self.dt)
        
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


class HestonGenerativeDataset(StochasticModelDataset):
    def __init__(
            self,
            n_paths=10000,
            n_steps=200,
            n_ts_features=1,
            s_price=100.0,
            mu=0.05,
            kappa=1.0,       # Mean reversion speed
            theta=0.04,      # Long-term variance
            sigma_v=0.1,     # Volatility of volatility
            rho=0.5,        # Correlation between asset and volatility
            v0=0.01,         # Initial variance
            T=1.0,
            return_log_returns=False,
            normalize=None,
    ):
        """
        Generates Heston model paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize)
        self.s_price = s_price
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        self.name = 'HestonGenerativeDataset'
        
        # Simulate Heston paths
        self.paths = self.simulate_paths()
        self.process_paths()

    def simulate_paths(self):
        """
        Simulates Heston model paths.
        """
        return self.simulate_heston_paths(
            self.n_paths, self.n_steps-1, self.s_price, self.mu, 
            self.kappa, self.theta, self.sigma_v, self.rho, self.v0, self.dt
        )
        
    def simulate_heston_paths(self, n_paths, n_steps, S0, mu, kappa, theta, sigma_v, rho, v0, dt):
        """
        Simulates Heston model paths with stochastic volatility.
        
        Parameters:
        - kappa: Mean reversion speed for variance process
        - theta: Long-term variance
        - sigma_v: Volatility of volatility
        - rho: Correlation between asset returns and variance
        - v0: Initial variance
        """
        # Initialize arrays
        prices = np.zeros((n_paths, n_steps + 1))
        variances = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        prices[:, 0] = S0
        variances[:, 0] = v0
        
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, size=(n_paths, n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, size=(n_paths, n_steps))
        
        # Simulate paths
        for t in range(n_steps):
            # Ensure variance stays positive (apply reflection or truncation)
            variances[:, t] = np.maximum(variances[:, t], 0)
            
            # Update price using Euler scheme
            prices[:, t+1] = prices[:, t] * np.exp(
                (mu - 0.5 * variances[:, t]) * dt + 
                np.sqrt(variances[:, t] * dt) * Z1[:, t]
            )
            
            # Update variance using Euler scheme
            variances[:, t+1] = variances[:, t] + kappa * (theta - variances[:, t]) * dt + \
                               sigma_v * np.sqrt(variances[:, t] * dt) * Z2[:, t]
        self.variances = variances
        return prices



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

