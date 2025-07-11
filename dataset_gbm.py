import json

import numpy as np
import torch
from torch.utils.data import Dataset
import yfinance as yf
import pandas as pd
import hashlib
import joblib
import os
import time


class StochasticModelDataset(Dataset):
    def __init__(
        self,
        n_paths=10000,
        n_steps=200,
        n_ts_features=1,
        T=1.0,
        return_log_returns=False,
        normalize=None,
        corruption_probability=0.0,
        sigma=0.0,
        noise_type='ve',
        **kwargs
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
        self.T = T
        self.dt = T/n_steps
        self.corruption_probability = corruption_probability
        self.sigma = sigma
        self.noise_type = noise_type
        assert self.normalize in [None, 'global_zscore', 'per_path_zscore', 'global_mean']
        
        self.paths = None
        
        # unet compatibility
        self.resolution = self.n_steps
        self.num_channels = self.n_ts_features
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
        if self.paths.shape == (self.n_paths, self.n_steps):
            self.paths = np.expand_dims(self.paths, axis=1)
        assert self.paths.shape == (self.n_paths, self.n_ts_features, self.n_steps)
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

        sigma_n = None
        if np.random.rand() > self.corruption_probability:
            noise_level = 0.0
            noise = np.zeros_like(ts)
        else:

            if self.sigma > 0:
                if self.noise_type == 've':
                    noise = np.random.normal(size=ts.shape)
                    ts += self.sigma*noise
                elif self.noise_type == 've_gbm':
                    noise = np.random.normal(size=ts.shape)
                    sigma_n = np.sqrt((self.sigma/self.sigma_gbm)**2 - 1)
                    ts += sigma_n*noise
                elif self.noise_type == 'extra_noise':
                    noise = np.random.normal(size=ts.shape)
                    ts += self.sigma*noise
                else:
                    raise NotImplementedError
            else:
                noise = np.zeros_like(ts)
            if sigma_n is not None:
                noise_level = sigma_n
            else:
                noise_level = self.sigma
        if self.noise_type == 'denoise_only':
            noise_level = self.sigma
        if self.noise_type == 'extra_noise':
            noise_level = 0.0

        return {
            "image": ts.copy(),
            "label": np.zeros(0, dtype=np.float32),
            'sigma': noise_level,
            'noise': noise,
        }

class STDGaussianGenerativeDataset(StochasticModelDataset):
    def __init__(
            self,
            n_paths=10000,
            n_steps=256,
            n_ts_features=1,
            s_price=100.0,
            mu=0.05,
            sigma_gbm=0.2,
            T=1.0,
            return_log_returns=False,
            normalize=None,
            **kwargs
    ):
        """
        Generates GBM paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize, **kwargs)
        self.s_price = s_price
        self.mu = mu
        self.sigma_gbm = sigma_gbm
        self.name = 'GBMGenerativeDataset'
        
        # Simulate GBM paths
        self.paths = self.simulate_paths()
        self.resolution = self.n_steps

    def simulate_paths(self):
        """
        Simulates Geometric Brownian Motion (GBM) paths.
        """
        rng = np.random.default_rng(seed=999)
        return rng.normal(
            # scale=(1+0.0165**2)**0.5,
            size=(self.n_paths, self.n_ts_features, self.n_steps)
            ) 

    def __str__(self):
        """
        String representation of Geometric Brownian Motion model parameters.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Steps": self.n_steps,
                "Time Horizon (T)": self.T,
                "Time Step (dt)": round(self.dt, 4),
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "GBM Specific Parameters": {
                "Initial Stock Price (S0)": self.s_price,
                "Drift (mu)": self.mu,
                "Volatility (sigma)": self.sigma_gbm
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)
class GBMGenerativeDataset(StochasticModelDataset):
    def __init__(
            self,
            n_paths=10000,
            n_steps=200,
            n_ts_features=1,
            s_price=100.0,
            mu=0.05,
            sigma_gbm=0.2,
            T=1.0,
            return_log_returns=False,
            normalize=None,
            **kwargs
    ):
        """
        Generates GBM paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize, **kwargs)
        self.s_price = s_price
        self.mu = mu
        self.sigma_gbm = sigma_gbm
        self.name = 'GBMGenerativeDataset'
        
        # Simulate GBM paths
        self.paths = self.simulate_paths()
        self.process_paths()

    def simulate_paths(self):
        """
        Simulates Geometric Brownian Motion (GBM) paths.
        """
        return self.simulate_gbm_paths(self.n_paths, self.n_steps - 1, self.s_price, self.mu, self.sigma_gbm, self.dt)
        
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

    def __str__(self):
        """
        String representation of Geometric Brownian Motion model parameters.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Steps": self.n_steps,
                "Time Horizon (T)": self.T,
                "Time Step (dt)": round(self.dt, 4),
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "GBM Specific Parameters": {
                "Initial Stock Price (S0)": self.s_price,
                "Drift (mu)": self.mu,
                "Volatility (sigma)": self.sigma_gbm
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)


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
            **kwargs
    ):
        """
        Generates Heston model paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize, **kwargs)
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

    def __str__(self):
        """
        String representation of Heston model parameters.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Steps": self.n_steps,
                "Time Horizon (T)": self.T,
                "Time Step (dt)": round(self.dt, 4),
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "Heston Specific Parameters": {
                "Initial Stock Price (S0)": self.s_price,
                "Risk-Free Rate (mu)": self.mu,
                "Mean Reversion Speed (kappa)": self.kappa,
                "Long-Term Variance (theta)": self.theta,
                "Volatility of Volatility (sigma_v)": self.sigma_v,
                "Correlation (rho)": self.rho,
                "Initial Variance (v0)": self.v0
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)

class MJDGenerativeDataset(StochasticModelDataset):
    def __init__(
            self,
            n_paths=10000,
            n_steps=200,
            n_ts_features=1,
            s_price=100.0,
            mu=0.05,
            sigma_gbm=0.2,
            lamb=0.1,
            mu_j=0.1,
            sigma_j=0.1,
            T=1.0,
            return_log_returns=False,
            normalize=None,
            **kwargs
    ):
        """
        Generates Merton Jump Diffusion model paths with shape (n_paths, n_steps, n_ts_features).
        Can optionally return log returns instead of price paths.
        """
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize, **kwargs)
        self.s_price = s_price
        self.mu = mu
        self.sigma_gbm = sigma_gbm
        self.lamb = lamb
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.name = 'MJDGenerativeDataset'

        # Simulate MJD paths
        self.paths = self.simulate_paths()
        self.process_paths()

    def simulate_paths(self):
        """
        Simulates Merton Jump Diffusion model paths.
        """
        return self.simulate_mjd_paths(
            self.n_paths, self.n_steps-1, self.s_price, self.mu,
            self.sigma_gbm, self.lamb, self.mu_j, self.sigma_j, self.dt
        )

    def simulate_mjd_paths(self, n_paths, n_steps, S0, mu, sigma, lamb, mu_j, sigma_j, dt):
        """
        Simulates the Merton Jump Diffusion model.
        """
        size = (n_steps, n_paths)
        poi_rv = np.multiply(np.random.poisson(lamb * dt, size=size),
                             np.random.normal(mu_j, sigma_j, size=size)).cumsum(axis=0)
        geo = np.cumsum(((mu - sigma ** 2 / 2 - lamb * (mu_j + sigma_j ** 2 * 0.5)) * dt + \
                         sigma * np.sqrt(dt) * \
                         np.random.normal(size=size)), axis=0)

        res = np.exp(geo + poi_rv)
        res = np.vstack([np.ones((1, n_paths)), res])
        return res.T * S0



    def __str__(self):
        """
        String representation of Merton Jump Diffusion model parameters.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Steps": self.n_steps,
                "Time Horizon (T)": self.T,
                "Time Step (dt)": round(self.dt, 4),
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "MJD Specific Parameters": {
                "Initial Stock Price (S0)": self.s_price,
                "Risk-Free Rate (mu)": self.mu,
                "Volatility (sigma)": self.sigma_gbm,
                "Jump Intensity (lambda)": self.lamb,
                "Jump Mean (mu_j)": self.mu_j,
                "Jump Volatility (sigma_j)": self.sigma_j
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)

class RealMarketDataset(StochasticModelDataset):
    def __init__(self, symbol, sliding_window='non_overlapping', **kwargs):
        """
        Loads real market data for a given symbol.
        """
        super().__init__(**kwargs)
        self.symbol = symbol
        self.sliding_window = sliding_window
        self.paths = self.load_data()
        self.process_paths()

        self.name = 'RealMarketDataset'

    def load_data(self):
        """
        Loads real market data for a given symbol.
        """
        data = yf.download(self.symbol, period='max')
        data = data['Close'].dropna().to_numpy().squeeze()
        return data

    def process_paths(self):

        if self.return_log_returns:
            paths = self.compute_log_returns()
        else:
            paths = self.paths

        if self.sliding_window == 'non_overlapping':
            paths = self.non_overlapping_swv(paths, self.n_steps)
        elif self.sliding_window == 'overlapping':
            paths = np.lib.stride_tricks.sliding_window_view(paths, self.n_steps)
        elif self.sliding_window == 'max':
            paths = self.paths.reshape(1, self.n_steps)

        self.n_paths = paths.shape[0]
        self.paths = np.expand_dims(paths, axis=1)
        assert self.paths.shape == (self.n_paths, 1, self.n_steps)
        self.resolution = self.n_steps
        self.mean = np.mean(self.paths)
        self.std = np.std(self.paths)

    def __str__(self):
        """
        String representation of Real Market dataset.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Steps": self.n_steps,
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "Real Market Data": {
                "Symbol": self.symbol
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)

    @staticmethod
    def non_overlapping_swv(arr, window_size):
        """
        Compute non-overlapping sliding window view of an array.
        """
        max_length = arr.shape[0] - arr.shape[0] % window_size
        return arr[:max_length].reshape(-1, window_size)


class HistoricalMarketDataset(StochasticModelDataset):
    def __init__(self, start='2024-01-01', end='2025-01-03', symbols=None, **kwargs):
        """
        Loads real market data for a given symbol.
        """
        super().__init__(**kwargs)
        if symbols is None:
            symbols = self.get_symbols()
        self.symbols = symbols
        self.start = start
        self.end = end 
        self.paths = self.load_data()
        self.process_paths()

        self.name = 'HistoricalMarketDataset'

    def get_cache_filename(self):
        """
        Generates a unique cache filename based on symbols and date range.
        """
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{self.symbols}_{self.start}_{self.end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(cache_dir, f"{hash_key}.pkl")

    def load_data(self, batch_size=1000, pause_time=5):
        """
        Loads real market data for a given symbol (cached). Downloads in batches to avoid rate limits.
        """
        cache_file = self.get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading data from cache...")
            data = joblib.load(cache_file)
        else:
            print("Downloading data from yfinance in batches...")
            all_data = []

            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i + batch_size]
                print(f"Downloading batch {i} to {i + batch_size}...")
                try:
                    batch_data = yf.download(
                        batch,
                        start=self.start,
                        end=self.end,
                        threads=False,
                    )
                    all_data.append(batch_data)
                except Exception as e:
                    print(f"Batch {i} failed: {e}")
                time.sleep(pause_time)

            # Combine all downloaded data
            data = pd.concat(all_data, axis=1)
            joblib.dump(data, cache_file)

        close_prices = data['Close']
        bad_tickers = close_prices.columns[(close_prices < 0).any()]
        print(f"\nRemoving bad tickers: {bad_tickers.tolist()}\n")

        clean_close = close_prices.drop(columns=bad_tickers)
        return clean_close.interpolate(method='time').ffill().bfill().dropna(axis=1).to_numpy().T

    def process_paths(self):

        if self.return_log_returns:
            paths = self.compute_log_returns()
        else:
            paths = self.paths


        self.n_paths = paths.shape[0]
        self.n_steps = paths.shape[1]
        print('\n')
        print(paths.shape)
        print('\n')
        self.paths = np.expand_dims(paths, axis=1)
        assert self.paths.shape == (self.n_paths, 1, self.n_steps)
        self.resolution = self.n_steps
        self.mean = np.mean(self.paths)
        self.std = np.std(self.paths)

    def __str__(self):
        """
        String representation of Real Market dataset.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Steps": self.n_steps,
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "Real Market Data": {
                "# Symbols": len(self.symbols),
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)
    
    @staticmethod
    def get_symbols():
        df = pd.read_csv('us_symbols.csv')
        symbols = df['ticker'] 
        return symbols.astype(str).tolist()
        
class CorrelatedGBMGenerativeDataset(StochasticModelDataset):
    def __init__(
        self,
        n_paths=10000,
        n_steps=200,
        n_ts_features=3,
        s_price=100.0,
        mu=0.05,
        sigma_gbm=0.2,
        corr_matrix=None,
        T=1.0,
        return_log_returns=False,
        normalize=None,
        **kwargs
    ):
        super().__init__(n_paths, n_steps, n_ts_features, T, return_log_returns, normalize, **kwargs)
        self.s_price = s_price
        self.mu = self._expand_param(mu, n_ts_features)
        self.sigma_gbm = self._expand_param(sigma_gbm, n_ts_features)
        self.name = "CorrelatedGBMGenerativeDataset"

        if corr_matrix is None:
            self.corr_matrix = np.eye(n_ts_features)
        else:
            corr_matrix = np.asarray(corr_matrix)
            assert corr_matrix.shape == (n_ts_features, n_ts_features), "Invalid correlation matrix shape"
            self.corr_matrix = corr_matrix

        self.L = np.linalg.cholesky(self.corr_matrix)

        self.paths = self.simulate_paths()
        self.process_paths()

    def _expand_param(self, param, n):
        return np.full(n, param) if np.isscalar(param) else np.asarray(param)

    def simulate_paths(self):
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        # Sample independent normals: (n_paths, n_steps - 1, n_ts_features)
        Z = np.random.normal(size=(self.n_paths, self.n_steps - 1, self.n_ts_features))

        # Correlated Brownian increments
        correlated_increments = np.einsum("ij,pnj->pni", self.L, Z) * sqrt_dt

        # Expand mu and sigma: (1, 1, n_ts_features)
        mu = self.mu.reshape(1, 1, -1)
        sigma = self.sigma_gbm.reshape(1, 1, -1)

        # Calculate GBM log-returns
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * correlated_increments
        log_returns = drift + diffusion  # Shape: (n_paths, n_steps - 1, n_ts_features)

        # Insert zeros at start and exponentiate cumulative sum
        log_returns = np.concatenate([np.zeros((self.n_paths, 1, self.n_ts_features)), log_returns], axis=1)
        paths = np.exp(np.cumsum(log_returns, axis=1)) * self.s_price

        return paths.transpose(0, 2, 1)  # Final shape: (n_paths, n_ts_features, n_steps)

    def __str__(self):
        return json.dumps({
            "Model Name": self.name,
            "Paths": self.n_paths,
            "Steps": self.n_steps,
            "Features": self.n_ts_features,
            "Initial Price": self.s_price,
            "Mu": str(self.mu.tolist()),
            "Sigma": str(self.sigma_gbm.tolist()),
            # "Correlation Matrix": self.corr_matrix.tolist()
        }, indent=2)


class HistoricalMarketDatasetMultivar(StochasticModelDataset):
    def __init__(self, start='2024-01-01', end='2025-01-03', symbols=None, **kwargs):
        """
        Loads real market data for a given symbol.
        """
        super().__init__(**kwargs)
        if symbols is None:
            symbols = self.get_symbols()
        self.symbols = symbols
        self.start = start
        self.end = end 
        self.paths = self.load_data()
        self.process_paths()

        self.name = 'HistoricalMarketDataset'

    def get_cache_filename(self):
        """
        Generates a unique cache filename based on symbols and date range.
        """
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{self.symbols}_{self.start}_{self.end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(cache_dir, f"{hash_key}.pkl")

    def load_data(self, batch_size=5, pause_time=5):
        """
        Loads real market data for a given symbol (cached). Downloads in batches to avoid rate limits.
        Excludes 'Volume' column and filters out tickers with any negative price value.
        """
        cache_file = self.get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading data from cache...")
            data = joblib.load(cache_file)
        else:
            print("Downloading data from yfinance in batches...")
            all_data = []

            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i + batch_size]
                print(f"Downloading batch {i} to {i + batch_size}...")
                try:
                    batch_data = yf.download(
                        batch,
                        start=self.start,
                        end=self.end,
                        threads=False,
                        group_by='ticker'  # Ensures MultiIndex columns
                    )
                    all_data.append(batch_data)
                except Exception as e:
                    print(f"Batch {i} failed: {e}")
                time.sleep(pause_time)

            # Combine all downloaded data
            data = pd.concat(all_data, axis=1)
            joblib.dump(data, cache_file)

        # Drop 'Volume' column
        data = data.drop(columns=[col for col in data.columns if col[1] == 'Volume'])

        # Identify bad tickers with any price < 0 in any column
        bad_tickers = set()
        for ticker in data.columns.get_level_values(0).unique():
            subdf = data[ticker]
            if (subdf < 0).any().any():
                bad_tickers.add(ticker)

        print(f"\nRemoving bad tickers: {sorted(bad_tickers)}\n")
        data = data.drop(columns=bad_tickers, level=0)

        # Interpolate and return clean data
        data = data.interpolate(method='time').ffill().bfill().dropna(axis=1)
        
        # Convert from (time, ticker × feature) -> (ticker, feature, time)
        tickers = sorted(data.columns.get_level_values(0).unique())
        features = ['Open', 'High', 'Low', 'Close']

        # Ensure correct column order (ticker-major then feature)
        data = data.loc[:, [(ticker, feat) for ticker in tickers for feat in features if (ticker, feat) in data.columns]]

        # Convert to numpy and reshape
        n_timesteps = data.shape[0]
        n_tickers = len(tickers)
        n_features = len(features)

        data_np = data.to_numpy().reshape(n_timesteps, n_tickers, n_features)  # (time, tickers, features)
        return data_np.transpose(1, 2, 0)  # (tickers, features, time)


    def process_paths(self):

        if self.return_log_returns:
            paths = self.compute_log_returns()
        else:
            paths = self.paths


        self.n_paths = paths.shape[0]
        self.n_ts_features = paths.shape[1]
        self.n_steps = paths.shape[2]
        self.paths = paths
        assert self.paths.shape == (self.n_paths, self.n_ts_features, self.n_steps)
        self.resolution = self.n_steps
        self.mean = np.mean(self.paths)
        self.std = np.std(self.paths)

    def __str__(self):
        """
        String representation of Real Market dataset.
        """
        params = {
            "Model Name": self.name,
            "Path Generation Parameters": {
                "Number of Paths": self.n_paths,
                "Number of Features": self.n_ts_features,
                "Number of Steps": self.n_steps,
                "Return Log Returns": self.return_log_returns,
                "Normalization": self.normalize
            },
            "Real Market Data": {
                "# Symbols": len(self.symbols),
            },
            "Extra noise": {
                "sigma": self.sigma,
                "corr prob": self.corruption_probability,
                "noise type": self.noise_type,
            }
        }

        return json.dumps(params, indent=2)
    
    @staticmethod
    def get_symbols():
        df = pd.read_csv('us_symbols.csv')
        symbols = df['ticker'] 
        return symbols.astype(str).tolist()


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

def estimate_mjd_parameters(series, dt, threshold=3):
    """
    Estimate parameters of the Merton Jump Diffusion model from a price series.

    Parameters:
    - series: Time series of asset prices.
    - dt: Time step size.
    - threshold: Threshold (in standard deviations) to identify jumps.

    Returns:
    - params: Dictionary containing estimated parameters.
    """
    # Calculate log returns
    log_returns = np.diff(np.log(series))
    n = len(log_returns)

    # Calculate statistics of log returns
    mu_hat = np.mean(log_returns)
    sigma_hat = np.std(log_returns)

    # Identify jumps
    jump_indices = np.where(np.abs(log_returns - mu_hat) > threshold * sigma_hat)[0]
    no_jump_indices = np.where(np.abs(log_returns - mu_hat) <= threshold * sigma_hat)[0]

    # Estimate jump intensity (lambda)
    lamb_hat = len(jump_indices) / (n * dt)

    # Estimate jump sizes
    if len(jump_indices) > 0:
        jump_sizes = log_returns[jump_indices]
        mu_j_hat = np.mean(jump_sizes)
        sigma_j_hat = np.std(jump_sizes)
    else:
        mu_j_hat = 0
        sigma_j_hat = 0

    # Adjusted drift estimation
    # Remove jumps to estimate diffusion component
    diffusion_returns = log_returns[no_jump_indices]
    mu_diffusion_hat = np.mean(diffusion_returns) / dt
    sigma_diffusion_hat = np.std(diffusion_returns) / np.sqrt(dt)

    # Adjust drift for jump component
    k_hat = np.exp(mu_j_hat + 0.5 * sigma_j_hat**2) - 1
    mu_hat_adj = mu_diffusion_hat + lamb_hat * k_hat

    params = {
        "mu": mu_hat_adj,
        "sigma": sigma_diffusion_hat,
        "lamb": lamb_hat,
        "mu_j": None if len(jump_indices) == 0 else mu_j_hat,
        "sigma_j": None if len(jump_indices) <= 1 else sigma_j_hat,
    }

    return params