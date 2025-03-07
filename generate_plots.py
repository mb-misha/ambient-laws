import argparse
import json
import os

from dataset_gbm import GBMGenerativeDataset, estimate_parameters, reverse_log_return, HestonGenerativeDataset
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm

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


def process_data(generated_filepath, normalization, stochastic_model, mu, sigma, theta, v0, n_steps, return_log_returns, n_training_paths, n_paths=100000, K=110):
    outdir = os.path.dirname(generated_filepath)
    # Generate dataset
    if stochastic_model == 'GBM':
        dataset = GBMGenerativeDataset(
            n_paths=n_paths,
            n_steps=n_steps,
            mu=mu,
            sigma_gbm=sigma,
            return_log_returns=return_log_returns,
            normalize=normalization,
        )
    else:
        dataset = HestonGenerativeDataset(
            mu=mu,
            n_steps=n_steps,
            n_paths=n_paths,
            theta=theta,
            v0=v0,
            return_log_returns=return_log_returns,
            normalize=normalization,
        )
    real = dataset.paths.squeeze()
    generated = np.load(generated_filepath)
    if normalization == 'global_mean':
        generated_unnorm = generated * np.mean(real)
    elif normalization == 'global_zscore':
        generated_unnorm = generated * np.std(real) + np.mean(real)
    elif normalization is None:
        generated_unnorm = generated
    else:
        raise NotImplemented
    # Reverse log returns to obtain price paths
    if return_log_returns:
        real_gbm = reverse_log_return(real, s0=100)
        generated_gbm = reverse_log_return(generated_unnorm, s0=100)
    else:
        real_gbm = real
        generated_gbm = generated_unnorm
    # Estimate parameters
    mu_real, sigma_real = estimate_parameters(real_gbm, dataset.dt)
    mu_gen, sigma_gen = estimate_parameters(generated_gbm, dataset.dt)

    # Price options
    estimated_price_real, std_err_real, _ = price_option(paths=real_gbm.T, K=K, r=mu, T=1.0, M=n_paths)
    estimated_price_gen, std_err_gen, _ = price_option(paths=generated_gbm.T, K=K, r=mu, T=1.0, M=n_paths)
    if stochastic_model == 'GBM':
        bs_price = price_option_bs(S0=100, K=K, r=mu, sigma=sigma, T=1.0)


    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    # Plot 1: Real Paths
    for path in real[:100]:
        axes[0, 0].plot(path, alpha=0.5)
    axes[0, 0].set_title("Real Paths")

    # Plot 2: Generated Paths
    for path in generated_unnorm[:100]:
        axes[0, 1].plot(path, alpha=0.5)
    axes[0, 1].set_title("Generated Paths")

    # Plot 3: Real GBM Paths
    for path in real_gbm[:100]:
        axes[1, 0].plot(path, alpha=0.5)
    axes[1, 0].set_title("Real GBM Paths")

    # Plot 4: Generated GBM Paths
    for path in generated_gbm[:100]:
        axes[1, 1].plot(path, alpha=0.5)
    axes[1, 1].set_title("Generated GBM Paths")

    # Plot 5: KDE of Estimated Drift (Mu)
    sns.kdeplot(mu_real, label='real mu', fill=True, ax=axes[2, 0])
    sns.kdeplot(mu_gen, label='generated mu', fill=True, ax=axes[2, 0])
    # axes[2, 0].set_xlim(-0.2, 1.0)
    axes[2, 0].legend()
    axes[2, 0].set_title("KDE of Estimated Drift (Mu)")

    # Plot 6: KDE of Estimated Volatility (Sigma)
    sns.kdeplot(sigma_real, label='real sigma', fill=True, ax=axes[2, 1])
    sns.kdeplot(sigma_gen, label='generated sigma', fill=True, ax=axes[2, 1])
    # axes[2, 1].set_xlim(0, 0.2)
    axes[2, 1].legend()
    axes[2, 1].set_title("KDE of Estimated Volatility (Sigma)")

    # Plot 7: KDE of Log Returns
    sns.kdeplot(real.flatten(), label='real', fill=True, ax=axes[3, 0])
    sns.kdeplot(generated_unnorm.flatten(), label='generated', fill=True, ax=axes[3, 0])
    axes[3, 0].legend()
    axes[3, 0].set_title("KDE of Log Returns")

    params_str = str(dataset) + "\n" + f"K={K}\n" + f"Training set size: ({n_training_paths}, {n_steps-1})\n\n"

    if stochastic_model == 'Heston':
        option_pricing_str = (
               f"Real Option Price={estimated_price_real:.3f} ± {1.96 * std_err_real:.3f}\n"
               f"Generated Option Price={estimated_price_gen:.3f} ± {1.96 * std_err_gen:.3f}"
        )
    else:
        option_pricing_str = (
            f"BS Price={bs_price:.3f}\n"
            f"Real Option Price={estimated_price_real:.3f} ± {1.96 * std_err_real:.3f}\n"
            f"Generated Option Price={estimated_price_gen:.3f} ± {1.96 * std_err_gen:.3f}"
        )

    axes[3, 1].text(0.5, 0.5, params_str+option_pricing_str, fontsize=12, ha='center', va='center', bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    axes[3, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'paths_analysis.png'))

    if stochastic_model == 'Heston':
        realized_volatility = np.sqrt(np.mean(real**2, axis=0)*n_steps)
        realized_volatility_generated = np.sqrt(np.mean(generated_unnorm**2, axis=0)*n_steps)
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(dataset.variances, axis=0), label='True Variance (Heston)')
        plt.plot(realized_volatility**2, label='Realized Volatility (Heston)')
        plt.plot(realized_volatility_generated**2, label='Realized Volatility (Generated)')
        plt.legend()
        plt.title('True vs. estimated volatility (Heston) vs estimated volatility (generated)')
        plt.xlabel('Time Steps')
        plt.ylabel('Volatility')
        plt.savefig(os.path.join(outdir, 'volatility_analysis.png'))



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
        n_steps=dataset_args['n_steps'],
        return_log_returns=dataset_args['return_log_returns'],
        n_training_paths=dataset_args['n_paths'],
    )


