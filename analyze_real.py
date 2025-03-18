import argparse
import json
import os

from dataset_gbm import GBMGenerativeDataset, estimate_parameters, reverse_log_return, HestonGenerativeDataset, RealMarketDataset
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import skew, kurtosis

plt.style.use('seaborn-v0_8')



def process_data(generated_filepath, normalization, n_steps, return_log_returns, n_training_paths, noise_sigma, symbol, n_paths=100000, **kwargs):
    outdir = os.path.dirname(generated_filepath)
    # Generate dataset
    dataset = RealMarketDataset(
        symbol=symbol,
        n_steps=n_steps,
        return_log_returns=return_log_returns,
        normalize=normalization,
        sigma=noise_sigma,
    )
    real = dataset.paths.squeeze()
    generated = np.load(generated_filepath)
    if normalization == 'global_zscore':
        generated_unnorm = generated * dataset.std + dataset.mean
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


    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    # Plot 1: Real Paths
    #100 random indices
    indices = np.random.choice(real.shape[0], 100)

    for path in real[indices]:
        axes[0, 0].plot(path, alpha=0.5)
    axes[0, 0].set_title("Real Paths")

    # Plot 2: Generated Paths
    for path in generated_unnorm[indices]:
        axes[0, 1].plot(path, alpha=0.5)
    axes[0, 1].set_title("Generated Paths")

    # Plot 3: Real GBM Paths
    for path in real_gbm[indices]:
        axes[1, 0].plot(path, alpha=0.5)
    axes[1, 0].set_title("Real GBM Paths")

    # Plot 4: Generated GBM Paths
    for path in generated_gbm[indices]:
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

    params_str = str(dataset) + "\n" + f"Training set size: ({n_training_paths}, {n_steps-1})\n\n"

    axes[3, 1].text(0.5, 0.5, params_str, fontsize=12, ha='center', va='center', bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    axes[3, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'paths_analysis.png'))

    realized_volatility = np.sqrt(np.mean(real**2, axis=0)*n_steps)
    realized_volatility_generated = np.sqrt(np.mean(generated_unnorm**2, axis=0)*n_steps)
    plt.figure(figsize=(10, 5))
    plt.plot(realized_volatility**2, label='Realized Volatility (Real)')
    plt.plot(realized_volatility_generated**2, label='Realized Volatility (Generated)')
    plt.legend()
    plt.title('True vs. estimated volatility vs estimated volatility (generated)')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.savefig(os.path.join(outdir, 'volatility_analysis.png'))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # KDE plot of log returns mean
    sns.kdeplot(np.mean(real, axis=1), fill=True, label='Real', ax=axs[0, 0])
    sns.kdeplot(np.mean(generated_unnorm, axis=1), fill=True, label='Generated', ax=axs[0, 0])
    axs[0, 0].set_title('KDE Plot of log returns mean')
    axs[0, 0].set_xlabel('mean')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()

    # KDE plot of log returns variance
    sns.kdeplot(np.var(real, axis=1), fill=True, label='Real', ax=axs[0, 1])
    sns.kdeplot(np.var(generated_unnorm, axis=1), fill=True, label='Synth', ax=axs[0, 1])
    axs[0, 1].set_title('KDE Plot of log returns variance')
    axs[0, 1].set_xlabel('variance')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()

    # KDE plot of skew
    sns.kdeplot(skew(real, axis=1), fill=True, label='Real', ax=axs[1, 0])
    sns.kdeplot(skew(generated_unnorm, axis=1), fill=True, label='Synth', ax=axs[1, 0])
    axs[1, 0].set_title('KDE Plot of skew')
    axs[1, 0].set_xlabel('skew value')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()

    # KDE plot of kurtosis
    sns.kdeplot(kurtosis(real, axis=1), fill=True, label='Real', ax=axs[1, 1])
    sns.kdeplot(kurtosis(generated_unnorm, axis=1), fill=True, label='Synth', ax=axs[1, 1])
    axs[1, 1].set_title('KDE Plot of kurtosis')
    axs[1, 1].set_xlabel('kurtosis value')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'log_returns_analysis.png'))

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
        n_steps=dataset_args['n_steps'],
        return_log_returns=dataset_args['return_log_returns'],
        n_training_paths=dataset_args['n_paths'],
        noise_sigma=dataset_args['sigma'],
        corruption_probability=dataset_args['corruption_probability'],
        noise_type=dataset_args['noise_type'],
        symbol=dataset_args['symbol'],
    )


