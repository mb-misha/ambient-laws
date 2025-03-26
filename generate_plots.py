import argparse
import json
import os

from matplotlib.backends.backend_pdf import PdfPages

from dataset_gbm import GBMGenerativeDataset, estimate_parameters, reverse_log_return, HestonGenerativeDataset, MJDGenerativeDataset
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
from heston_closed_form_solution import heston_price
from MJD_closed_form_solution import merton_jump_diffusion_price
import typing

plt.style.use('seaborn-v0_8')


def price_option(paths, K, r, T, M):
    terminal_prices = paths[-1, :]
    payoffs = np.maximum(terminal_prices - K, 0)
    option_prices = payoffs * np.exp(-r * T)
    estimated_price = np.mean(option_prices)
    std_err = np.std(option_prices) / np.sqrt(M)
    return estimated_price, std_err, option_prices

def price_option_bs(S0, K, r, sigma, T):

    # Black-Scholes formula components
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the call option price using the Black-Scholes formula
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def create_generative_dataset(
        stochastic_model: str,
        n_paths: int,
        n_steps: int,
        n_ts_features: int = 1,
        s_price: float = 100.0,
        mu: float = 0.05,
        sigma: float = None,
        theta: float = None,
        v0: float = None,
        # MJD specific parameters
        lamb: float = 0.1,
        mu_j: float = 0.1,
        sigma_j: float = 0.1,
        T: float = 1.0,
        return_log_returns: bool = False,
        normalization: str = None,
        noise_sigma: float = None,
        **kwargs
) -> typing.Tuple[np.ndarray, typing.Any]:
    """
    Create a generative dataset based on the specified stochastic model.

    Args:
        stochastic_model (str): Type of stochastic model ('GBM', 'Heston', or 'MJD')
        n_paths (int): Number of paths to generate
        n_steps (int): Number of steps in each path
        n_ts_features (int, optional): Number of time series features
        s_price (float, optional): Starting price for the asset
        mu (float): Mean drift parameter
        sigma (float, optional): Volatility for GBM/MJD
        theta (float, optional): Mean reversion parameter for Heston
        v0 (float, optional): Initial variance for Heston
        lamb (float, optional): Jump intensity for MJD
        mu_j (float, optional): Mean of jump size for MJD
        sigma_j (float, optional): Volatility of jump size for MJD
        T (float, optional): Total time horizon
        return_log_returns (bool, optional): Whether to return log returns
        normalization (str, optional): Normalization method
        noise_sigma (float, optional): Noise standard deviation
        **kwargs: Additional model-specific parameters

    Returns:
        Tuple containing generated paths and the dataset object
    """
    if stochastic_model == 'GBM':
        dataset = GBMGenerativeDataset(
            n_paths=n_paths,
            n_steps=n_steps,
            n_ts_features=n_ts_features,
            s_price=s_price,
            mu=mu,
            sigma_gbm=sigma,
            T=T,
            return_log_returns=return_log_returns,
            normalize=normalization,
            sigma=noise_sigma,
            **kwargs
        )
    elif stochastic_model == 'Heston':
        dataset = HestonGenerativeDataset(
            n_paths=n_paths,
            n_steps=n_steps,
            n_ts_features=n_ts_features,
            s_price=s_price,
            mu=mu,
            T=T,
            theta=theta,
            v0=v0,
            return_log_returns=return_log_returns,
            normalize=normalization,
            sigma=noise_sigma,
            **kwargs
        )
    elif stochastic_model == 'MJD':
        dataset = MJDGenerativeDataset(
            n_paths=n_paths,
            n_steps=n_steps,
            n_ts_features=n_ts_features,
            s_price=s_price,
            mu=mu,
            sigma_gbm=sigma,
            lamb=lamb,
            mu_j=mu_j,
            sigma_j=sigma_j,
            T=T,
            return_log_returns=return_log_returns,
            normalize=normalization,
            sigma=noise_sigma,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported stochastic model: {stochastic_model}")

    return dataset.paths.squeeze(), dataset


def unnormalize_data(
        generated: np.ndarray,
        real: np.ndarray,
        normalization: str = None
) -> np.ndarray:
    """
    Unnormalize generated data based on the specified normalization method.

    Args:
        generated (np.ndarray): Generated data
        real (np.ndarray): Real data for reference
        normalization (str, optional): Normalization method

    Returns:
        np.ndarray: Unnormalized generated data
    """
    if normalization == 'global_mean':
        return generated * np.mean(real)
    elif normalization == 'global_zscore':
        return generated * np.std(real) + np.mean(real)
    elif normalization is None:
        return generated
    else:
        raise NotImplementedError(f"Normalization method {normalization} not implemented")


def compute_option_pricing_results(
        real_paths: np.ndarray,
        generated_paths: np.ndarray,
        dataset,
        stochastic_model: str,
        mu: float,
        sigma: float = None,
        theta: float = None,
        v0: float = None,
        # MJD specific parameters
        lamb: float = 0.1,
        mu_j: float = 0.1,
        sigma_j: float = 0.1,
        n_paths: int = 100000,
        **kwargs
) -> pd.DataFrame:
    """
    Compute option pricing results for different strike prices.

    Args:
        ... (previous arguments)
        lamb (float, optional): Jump intensity for MJD
        mu_j (float, optional): Mean of jump size for MJD
        sigma_j (float, optional): Volatility of jump size for MJD
        ... (rest of previous arguments)

    Returns:
        pd.DataFrame: Option pricing results
    """
    results = []
    for K in range(70, 131, 10):
        # Compute real option price
        estimated_price_real, std_err_real, _ = price_option(
            paths=real_paths.T,
            K=K,
            r=mu,
            T=1.0,
            M=n_paths
        )

        # Compute generated option prices
        generated_paths_reshaped = generated_paths.reshape(-1, n_paths, real_paths.shape[-1])
        estimated_prices_gen_batch = []

        for batch in generated_paths_reshaped:
            estimated_price_gen, _, _ = price_option(
                paths=batch.T,
                K=K,
                r=mu,
                T=1.0,
                M=n_paths
            )
            estimated_prices_gen_batch.append(estimated_price_gen)

        estimated_prices_gen_avg = np.mean(estimated_prices_gen_batch)
        estimated_prices_gen_std_err = np.std(estimated_prices_gen_batch) / np.sqrt(len(estimated_prices_gen_batch))

        # Compute theoretical price
        if stochastic_model == 'GBM':
            theoretical_price = price_option_bs(S0=100, K=K, r=mu, sigma=sigma, T=1.0)
        elif stochastic_model == 'Heston':
            theoretical_price = heston_price(
                S0=100,
                K=K,
                r=mu,
                T=1.0,
                v0=v0,
                kappa=kwargs.get('kappa'),
                theta=theta,
                sigma=kwargs.get('sigma_v'),
                rho=kwargs.get('rho')
            )
        elif stochastic_model == 'MJD':
            theoretical_price = merton_jump_diffusion_price(
                S0=100,
                K=K,
                r=mu,
                T=1.0,
                sigma=sigma,
                lamb=lamb,
                mu_j=mu_j,
                sigma_j=sigma_j
            )
        else:
            raise ValueError(f"Unsupported stochastic model: {stochastic_model}")

        results.append({
            "Strike Price": K,
            "Monte Carlo Price (Real)": round(estimated_price_real, 3),
            "Generated Price": round(estimated_prices_gen_avg, 3),
            "Theoretical Price": round(theoretical_price, 3),
            "Relative Error (%)": round(100 * (estimated_prices_gen_avg - estimated_price_real) / estimated_price_real,
                                        3),
            "Std Error (Real)": std_err_real,
            "Std Error (Generated)": estimated_prices_gen_std_err,
        })

    return pd.DataFrame(results)


def plot_paths_and_distributions(
        real_paths: np.ndarray,
        generated_paths: np.ndarray,
        real_gbm_paths: np.ndarray,
        generated_gbm_paths: np.ndarray,
        mu_real: np.ndarray,
        mu_gen: np.ndarray,
        sigma_real: np.ndarray,
        sigma_gen: np.ndarray,
        dataset,
        n_training_paths: int,
        n_steps: int,
        stochastic_model: str = None,
        output_dir: str = '.'
):
    """
    Create visualizations of paths, parameter distributions, and log returns.

    Args:
        real_paths (np.ndarray): Original real paths
        generated_paths (np.ndarray): Generated paths
        real_gbm_paths (np.ndarray): Real GBM paths
        generated_gbm_paths (np.ndarray): Generated GBM paths
        mu_real (np.ndarray): Real drift estimates
        mu_gen (np.ndarray): Generated drift estimates
        sigma_real (np.ndarray): Real volatility estimates
        sigma_gen (np.ndarray): Generated volatility estimates
        dataset: Dataset object
        n_training_paths (int): Number of training paths
        n_steps (int): Number of steps
        stochastic_model (str, optional): Stochastic model type
        output_dir (str, optional): Directory to save plots
    """




def plot_path_comparisons(real_paths, generated_paths, real_gbm_paths, generated_gbm_paths, axes):
    """Plot comparison of paths for original and generated datasets."""
    path_types = [
        (real_paths, "Real Paths"),
        (generated_paths, "Generated Paths"),
        (real_gbm_paths, "Real GBM Paths"),
        (generated_gbm_paths, "Generated GBM Paths")
    ]

    for (paths, title), ax in zip(path_types, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]):
        for path in paths[:100]:
            ax.plot(path, alpha=0.5)
        ax.set_title(title)


def plot_parameter_densities(mu_real, mu_gen, sigma_real, sigma_gen, axes):
    """Plot kernel density estimates for drift and volatility."""
    # Drift (Mu) KDE
    sns.kdeplot(mu_real, label='real mu', fill=True, ax=axes[2, 0])
    sns.kdeplot(mu_gen, label='generated mu', fill=True, ax=axes[2, 0])


    # Set x-axis limits for Mu plot
    axes[2, 0].set_xlim(-2, 2)

    # Add mean lines
    axes[2, 0].axvline(np.mean(mu_real), color='blue', linestyle='--',
                       label='real mu mean')
    axes[2, 0].axvline(np.mean(mu_gen), color='orange', linestyle='--',
                       label='generated mu mean')
    axes[2, 0].legend()
    axes[2, 0].set_title("KDE of Estimated Drift (Mu)")

    # Volatility (Sigma) KDE
    sns.kdeplot(sigma_real, label='real sigma', fill=True, ax=axes[2, 1])
    sns.kdeplot(sigma_gen, label='generated sigma', fill=True, ax=axes[2, 1])

    # Set x-axis limits for Sigma plot
    axes[2, 1].set_xlim(0, 1.0)
    # Add mean lines
    axes[2, 1].axvline(np.mean(sigma_real), color='blue', linestyle='--',
                       label='real sigma mean')
    axes[2, 1].axvline(np.mean(sigma_gen), color='orange', linestyle='--',
                       label='generated sigma mean')

    axes[2, 1].legend()
    axes[2, 1].set_title("KDE of Estimated Volatility (Sigma)")


def plot_log_returns_density(real_paths, generated_paths, axes):
    """Plot kernel density estimates for log returns."""
    sns.kdeplot(real_paths.flatten(), label='real', fill=True, ax=axes[3, 0])
    sns.kdeplot(generated_paths.flatten(), label='generated', fill=True, ax=axes[3, 0])
    axes[3, 0].legend()
    axes[3, 0].set_title("KDE of Log Returns")


def add_parameter_text(axes, dataset, n_training_paths, n_steps):
    """Add text with model parameters to the plot."""
    params_str = (
        f"{dataset}\n"
        f"Training set size: ({n_training_paths}, {n_steps - 1})"
    )
    axes[3, 1].text(
        0.5, 0.5, params_str,
        fontsize=12, ha='center', va='center',
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}
    )
    axes[3, 1].set_axis_off()


def plot_heston_volatility(dataset, real_paths, generated_paths):
    """
    Plot volatility analysis for Heston model.

    Args:
        dataset: Dataset object containing variance information
        real_paths (np.ndarray): Real paths
        generated_paths (np.ndarray): Generated paths

    Returns:
        matplotlib.figure.Figure: Volatility comparison figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Calculate realized volatilities
    realized_volatility = np.sqrt(np.mean(real_paths ** 2, axis=0) * real_paths.shape[1])
    realized_volatility_generated = np.sqrt(np.mean(generated_paths ** 2, axis=0) * generated_paths.shape[1])

    # Plot volatility comparisons
    ax.plot(np.mean(dataset.variances, axis=0), label='True Variance (Heston)')
    ax.plot(realized_volatility ** 2, label='Realized Volatility (Heston)')
    ax.plot(realized_volatility_generated ** 2, label='Realized Volatility (Generated)')

    # Customize plot
    ax.legend()
    ax.set_title('True vs. estimated volatility (Heston) vs estimated volatility (generated)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Volatility')
    ax.set_ylim(0, 0.5)

    # Return the figure instead of saving
    return fig


def process_data(
        generated_filepath: str,
        normalization: str = None,
        stochastic_model: str = 'GBM',
        mu: float = 0.0,
        sigma: float = None,
        theta: float = None,
        v0: float = None,
        # MJD specific parameters
        lamb: float = 0.1,
        mu_j: float = 0.1,
        sigma_j: float = 0.1,
        n_steps: int = 252,
        n_ts_features: int = 1,
        s_price: float = 100.0,
        T: float = 1.0,
        return_log_returns: bool = False,
        n_training_paths: int = 1000,
        noise_sigma: float = 0.1,
        n_paths: int = 100000,
        **kwargs
):
    """
    Process and analyze generated financial data paths.

    Args:
        ... (previous arguments)
        lamb (float, optional): Jump intensity for MJD
        mu_j (float, optional): Mean of jump size for MJD
        sigma_j (float, optional): Volatility of jump size for MJD
        n_ts_features (int, optional): Number of time series features
        s_price (float, optional): Starting price for the asset
        T (float, optional): Total time horizon
        ... (rest of previous arguments)
    """
    # Determine output directory
    outdir = os.path.dirname(generated_filepath)

    # Create generative dataset
    real_paths, dataset = create_generative_dataset(
        stochastic_model=stochastic_model,
        n_paths=n_paths,
        n_steps=n_steps,
        n_ts_features=n_ts_features,
        s_price=s_price,
        mu=mu,
        sigma=sigma,
        theta=theta,
        v0=v0,
        lamb=lamb,
        mu_j=mu_j,
        sigma_j=sigma_j,
        T=T,
        return_log_returns=return_log_returns,
        normalization=normalization,
        noise_sigma=noise_sigma,
        **kwargs
    )

    # Load generated data
    generated = np.load(generated_filepath)

    # Unnormalize generated data
    generated_unnorm = unnormalize_data(generated, real_paths, normalization)

    # Reverse log returns if needed
    if return_log_returns:
        real_gbm = reverse_log_return(real_paths, s0=s_price)
        generated_gbm = reverse_log_return(generated_unnorm, s0=s_price)
    else:
        real_gbm = real_paths
        generated_gbm = generated_unnorm

    # Estimate parameters
    mu_real, sigma_real = estimate_parameters(real_gbm, dataset.dt)
    mu_gen, sigma_gen = estimate_parameters(generated_gbm, dataset.dt)

    # Compute option pricing results
    results_df = compute_option_pricing_results(
        real_paths,
        generated_gbm,
        dataset,
        stochastic_model,
        mu,
        sigma,
        theta,
        v0,
        lamb=lamb,
        mu_j=mu_j,
        sigma_j=sigma_j,
        **kwargs
    )

    # Save results to CSV
    results_df.to_csv(os.path.join(outdir, "option_pricing_results.csv"), index=False)


    output_filename = os.path.join(outdir, 'paths_analysis.pdf')
    with PdfPages(output_filename) as pdf:
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))

        # Path visualization subplots
        plot_path_comparisons(real_paths, generated_unnorm, real_gbm, generated_gbm, axes)

        # Density estimation plots
        plot_parameter_densities(mu_real, mu_gen, sigma_real, sigma_gen, axes)

        # Log returns density plot
        plot_log_returns_density(real_paths, generated_unnorm, axes)

        # Parameter info text
        add_parameter_text(axes, dataset, n_training_paths, n_steps)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Additional Heston volatility plot if applicable
        if stochastic_model == 'Heston':
            heston_fig = plot_heston_volatility(dataset, real_paths, generated_unnorm)
            pdf.savefig(heston_fig)
            plt.close()
        fix, ax = plt.subplots()
        ax.axis('off')
        pd.plotting.table(ax, results_df, loc='center', colWidths=[0.1] * len(results_df.columns))
        pdf.savefig(fix)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_filepath", type=str, required=True)
    args = parser.parse_args()

    params_dir = os.path.dirname(os.path.dirname(args.generated_filepath))
    params_path = os.path.join(params_dir, 'training_options.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    dataset_args = params['gbm_kwargs']
    process_data(
        generated_filepath=args.generated_filepath,
        normalization=dataset_args['normalize'],
        stochastic_model=params['stochastic_model'],
        mu=dataset_args['mu'],
        sigma=dataset_args.get('sigma_gbm', None),
        theta=dataset_args.get('theta', None),
        v0=dataset_args.get('v0', None),
        rho=dataset_args.get('rho', None),
        sigma_v=dataset_args.get('sigma_v', None),
        kappa=dataset_args.get('kappa', None),
        n_steps=dataset_args['n_steps'],
        return_log_returns=dataset_args['return_log_returns'],
        n_training_paths=dataset_args['n_paths'],
        noise_sigma=dataset_args['sigma'],
        corruption_probability=dataset_args['corruption_probability'],
        noise_type=dataset_args['noise_type'],
        mu_j=dataset_args.get('mu_j', None),
        sigma_j=dataset_args.get('sigma_j', None),
        lamb=dataset_args.get('lamb', None),
    )


