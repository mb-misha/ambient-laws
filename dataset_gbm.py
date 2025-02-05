import math
import numpy as np
import torch
from torch.utils.data import Dataset



class GBMGenerativeDataset(Dataset):
    def __init__(
        self,
        n_paths=10000,
        n_steps=200,
        n_ts_features=1,
        S0=100.0,
        mu=0.05,
        sigma=0.2,
        dt=1/252
    ):
        """
        Generates GBM paths with shape (n_paths, n_steps, n_ts_features).
        """
        self.n_ts_features = n_ts_features
        self.paths = self.simulate_gbm_paths(n_paths, n_steps, S0, mu, sigma, dt)
        self.name = 'GBMGenerativeDataset'
        self.resolution = n_steps

    def simulate_gbm_paths(self, n_paths, n_steps, S0, mu, sigma, dt):
        paths = np.zeros((n_paths, n_steps, self.n_ts_features), dtype=np.float32)
        paths[:, 0, 0] = S0  # Initialize first step for all paths
        for i in range(1, n_steps):
            Z = np.random.normal(0, 1, n_paths)
            paths[:, i, 0] = paths[:, i - 1, 0] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
            )
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.tensor(self.paths[idx], dtype=torch.float32)
