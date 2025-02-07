import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

sns.set_style("whitegrid")  # Use seaborn styling

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
            return_log_returns=False,  # New parameter
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
        self.paths = self.simulate_gbm_paths(n_paths, n_steps, s_price, mu, sigma, dt)

        if self.return_log_returns:
            self.paths = self.compute_log_returns(self.paths)

        self.dt = dt  # Store dt for parameter estimation
        self.name = 'GBMGenerativeDataset'
        self.resolution = n_steps

    def simulate_gbm_paths(self, n_paths, n_steps, S0, mu, sigma, dt):
        """
        Simulates Geometric Brownian Motion (GBM) paths.
        """
        paths = np.zeros((n_paths, n_steps, self.n_ts_features), dtype=np.float32)
        paths[:, 0, 0] = S0  # Initialize first step for all paths

        for i in range(1, n_steps):
            Z = np.random.normal(0, 1, n_paths)
            paths[:, i, 0] = paths[:, i - 1, 0] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z
            )
        return paths

    def compute_log_returns(self, paths):
        """
        Computes log returns for each GBM path.
        Log return at time t is: log(S_t / S_{t-1}).
        """
        log_returns = np.zeros_like(paths, dtype=np.float32)
        log_returns[:, 1:, :] = np.log(paths[:, 1:, :] / paths[:, :-1, :])  # Compute log returns
        log_returns[:, 0, :] = 0  # Set first row to 0 since there's no previous value
        return log_returns

    def estimate_parameters(self):
        """
        Estimates the drift (μ) and volatility (σ) from the generated paths.
        """
        log_returns = self.compute_log_returns(self.paths) if not self.return_log_returns else self.paths

        # Compute μ and σ using dt for correct scaling
        mu_est = np.mean(log_returns[:, 1:, :]) / self.dt  # Scale drift to annualized
        sigma_est = np.std(log_returns[:, 1:, :]) / np.sqrt(self.dt)  # Scale volatility to annualized

        return mu_est, sigma_est

    def plot_subset(self, num_samples=50, save_path=None):
        """
        Plots a subset of the generated paths.

        Args:
            num_samples (int): Number of paths to plot.
            save_path (str, optional): File path to save the plot. If None, displays the plot.
        """
        subset_indices = np.random.choice(self.n_paths, min(num_samples, self.n_paths), replace=False)
        subset_paths = self.paths[subset_indices, :, 0]  # Select 50 paths

        plt.figure(figsize=(10, 5))

        for i, ts in enumerate(subset_paths):
            plt.plot(ts, alpha=0.7, label=f"Path {i+1}" if i < 5 else "_nolegend_")  # Show labels for first few

        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Value' if not self.return_log_returns else 'Log Returns', fontsize=12)
        plt.title(f'Subset of Generated {"Log Returns" if self.return_log_returns else "GBM"} Paths', fontsize=14)

        # Estimate μ and σ
        mu_est, sigma_est = self.estimate_parameters()
        plt.figtext(0.15, 0.8, f"Estimated μ: {mu_est:.4f}", fontsize=12, color="blue")
        plt.figtext(0.15, 0.75, f"Estimated σ: {sigma_est:.4f}", fontsize=12, color="red")

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, frameon=True)  # Move legend outside
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.tensor(self.paths[idx], dtype=torch.float32)
