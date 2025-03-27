import argparse
import json
import os

from dataset_gbm import GBMGenerativeDataset, estimate_parameters, reverse_log_return, HestonGenerativeDataset, RealMarketDataset, MJDGenerativeDataset
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy
import pandas as pd
from heston_closed_form_solution import heston_price
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.autolayout'] = True


def price_option_call(paths, K, r, T, M):
    terminal_prices = paths[-1, :]
    payoffs = np.maximum(terminal_prices - K, 0)
    option_prices = payoffs * np.exp(-r * T)
    estimated_price = np.mean(option_prices)
    std_err = np.std(option_prices) / np.sqrt(M)
    return estimated_price, std_err, option_prices

def price_option_put(paths, K, r, T, M):
    terminal_prices = paths[-1, :]
    payoffs = np.maximum(K - terminal_prices, 0)
    option_prices = payoffs * np.exp(-r * T)
    estimated_price = np.mean(option_prices)
    std_err = np.std(option_prices) / np.sqrt(M)
    return estimated_price, std_err, option_prices

def price_option_call_bs(S0, K, r, sigma, T):

    # Black-Scholes formula components
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the call option price using the Black-Scholes formula
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def price_option_put_bs(S0, K, r, sigma, T):
    # Black-Scholes formula components
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the put option price using the Black-Scholes formula
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def price_option_bs(S0, K, r, sigma, T, option_type='call'):
    if option_type == 'call':
        return price_option_call_bs(S0, K, r, sigma, T)
    elif option_type == 'put':
        return price_option_put_bs(S0, K, r, sigma, T)
    else:
        raise ValueError(f"Invalid option type: {option_type}")


def compute_var_cvar(returns, alpha=0.95):
    var = np.percentile(returns, 100*(1-alpha), axis=1)
    cvar_mask = returns <= var[:, np.newaxis]
    cvar = np.nanmean(np.where(cvar_mask, returns, np.nan), axis=1)
    return var, cvar




def process_data(generated_filepath, normalization, stochastic_model, mu, sigma, theta, v0, n_steps, return_log_returns, n_training_paths, noise_sigma, n_paths=100000, K=110, **kwargs):
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
            sigma=noise_sigma,
            **kwargs
        )
    elif stochastic_model == 'Heston':
        dataset = HestonGenerativeDataset(
            mu=mu,
            n_steps=n_steps,
            n_paths=n_paths,
            theta=theta,
            v0=v0,
            return_log_returns=return_log_returns,
            normalize=normalization,
            sigma=noise_sigma,
            **kwargs
        )
    elif stochastic_model == 'MJD':
        dataset = MJDGenerativeDataset(
            mu=mu,
            n_steps=n_steps,
            n_paths=n_paths,
            sigma_gbm=sigma,
            lamb=kwargs.get('lamb'),
            mu_j=kwargs.get('mu_j'),
            sigma_j=kwargs.get('sigma_j'),
            return_log_returns=return_log_returns,
            normalize=normalization,
            sigma=noise_sigma,
        )
    elif stochastic_model == 'MarketData':
        dataset = RealMarketDataset(
            symbol=kwargs.get('symbol'),
            n_steps=n_steps,
            return_log_returns=return_log_returns,
            normalize=normalization,
            sigma=noise_sigma,
        )
    else:
        raise ValueError(f"Invalid stochastic model: {stochastic_model}")

    real = dataset.paths.squeeze()
    generated = np.load(generated_filepath)
    if stochastic_model == 'MarketData':
        generated = generated[:real.shape[0], :]
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

    real_norm = (real - dataset.mean)/dataset.std


    raw_data_outdir = os.path.join(outdir, 'raw_data')
    os.makedirs(raw_data_outdir, exist_ok=True)
    output_filename = os.path.join(outdir, 'paths_analysis.pdf')
    with PdfPages(output_filename) as pdf:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Real vs. Generated Paths ({stochastic_model})")
        # Plot 1: Real Paths
        for path in real[:100]:
            axes[0, 0].plot(path, alpha=0.5)
        axes[0, 0].set_title("Training Paths")

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
        axes[1, 1].set_title("Generated GBM Paths (restored from log returns)")
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'gen_paths.png'))
        plt.close(fig)

        # Analyze normalized data
        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Distribution of Normalized Data')
        plot_distribution(axes, generated, real_norm)
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'kde_log_returns_norm.png'))
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Sample Statistics of Normalized Data')
        stats_df = plot_sample_statistics(axes, generated, real_norm)

        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'log_return_stats_norm.png'))
        plt.close(fig)
        stats_df.to_csv(os.path.join(raw_data_outdir, 'log_return_stats_norm.csv'))

        fig, axes = plt.subplots()
        fig.suptitle('Sample Statistics of Normalized Data')
        pd.plotting.table(axes, stats_df, loc='center')
        axes.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


        # Analyze denormalized data
        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Distribution of Denormalized Data')
        plot_distribution(axes, generated_unnorm, real)
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'kde_log_returns.png'))
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Sample Statistics of Denormalized Data')
        stats_df = plot_sample_statistics(axes, generated_unnorm, real)
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'log_return_stats.png'))
        plt.close(fig)
        stats_df.to_csv(os.path.join(raw_data_outdir, 'log_return_stats.csv'))

        fig, axes = plt.subplots()
        fig.suptitle('Sample Statistics of Denormalized Data')
        pd.plotting.table(axes, stats_df, loc='center')
        axes.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Autocorrelation of Log Returns')
        plot_acf(real[0], lags=50, ax=axes[0, 0])
        axes[0, 0].set_title("ACF of Real Log Returns")
        plot_acf(generated_unnorm[0], lags=50, ax=axes[0, 1])
        axes[0, 1].set_title("ACF of Generated Log Returns")

        plot_acf(real[0]**2, lags=50, ax=axes[1, 0])
        axes[1, 0].set_title("ACF of Real Log Returns Squared")
        plot_acf(generated_unnorm[0]**2, lags=50, ax=axes[1, 1])
        axes[1, 1].set_title("ACF of Generated Log Returns Squared")
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'acf_log_returns.png'))
        plt.close(fig)



        # # Plot autocorrelation
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # real_acf_values = np.array([sm.tsa.acf(path, nlags=50) for path in real])
        # generated_acf_values = np.array([sm.tsa.acf(path, nlags=50) for path in generated_unnorm])
        #
        # real_acf_mean = np.mean(real_acf_values, axis=0)
        # real_acf_std = np.std(real_acf_values, axis=0)
        # generated_acf_mean = np.mean(generated_acf_values, axis=0)
        # generated_acf_std = np.std(generated_acf_values, axis=0)
        #
        # real_lags = np.arange(len(real_acf_mean))
        # generated_lags = np.arange(len(generated_acf_mean))
        #
        # axes[0].plot(real_lags, real_acf_mean, label='real')
        # axes[0].fill_between(real_lags, real_acf_mean - 1.96*real_acf_std, real_acf_mean + 1.96*real_acf_std, alpha=0.3)
        # axes[0].set_title("Autocorrelation of Real Log Returns")
        #
        # axes[1].plot(generated_lags, generated_acf_mean, label='generated')
        # axes[1].fill_between(generated_lags, generated_acf_mean - 1.96*generated_acf_std, generated_acf_mean + 1.96*generated_acf_std, alpha=0.3)
        # axes[1].set_title("Autocorrelation of Generated Log Returns")
        # pdf.savefig(fig)
        # plt.savefig(os.path.join(raw_data_outdir, 'autocorrelation.png'))
        # plt.close(fig)


        # Plot GBM parameters
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('GBM Parameters Estimation')

        # Estimate parameters
        mu_real, sigma_real = estimate_parameters(real_gbm, dataset.dt)
        mu_gen, sigma_gen = estimate_parameters(generated_gbm, dataset.dt)
        mu_real_mean = np.mean(mu_real)
        mu_gen_mean = np.mean(mu_gen)
        sigma_real_mean = np.mean(sigma_real)
        sigma_gen_mean = np.mean(sigma_gen)

        # Plot 5: KDE of Estimated Drift (Mu)
        sns.kdeplot(mu_real, label='real mu', fill=True, ax=axes[0])
        sns.kdeplot(mu_gen, label='generated mu', fill=True, ax=axes[0])
        axes[0].axvline(mu_real_mean, color='blue', linestyle='--', label=f'real mu mean ({mu_real_mean:.3f})')
        axes[0].axvline(mu_gen_mean, color='orange', linestyle='--', label=f'generated mu mean ({mu_gen_mean:.3f})')
        axes[0].legend()
        axes[0].set_title("KDE of Estimated Drift (Mu)")

        # Plot 6: KDE of Estimated Volatility (Sigma)
        sns.kdeplot(sigma_real, label='real sigma', fill=True, ax=axes[1])
        sns.kdeplot(sigma_gen, label='generated sigma', fill=True, ax=axes[1])
        axes[1].axvline(sigma_real_mean, color='blue', linestyle='--', label=f'real sigma mean ({sigma_real_mean:.3f})')
        axes[1].axvline(sigma_gen_mean, color='orange', linestyle='--', label=f'generated sigma mean ({sigma_gen_mean:.3f})')
        axes[1].legend()
        axes[1].set_title("KDE of Estimated Volatility (Sigma)")
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'gbm_parameters.png'))
        plt.close(fig)


        if stochastic_model == 'Heston':
            fig, ax = plt.subplots()
            fig.suptitle('Heston Volatility Analysis')
            realized_volatility = np.sqrt(np.mean(real**2, axis=0)*n_steps)
            realized_volatility_generated = np.sqrt(np.mean(generated_unnorm**2, axis=0)*n_steps)
            ax.plot(np.mean(dataset.variances, axis=0), label='True Variance (Heston)')
            ax.plot(realized_volatility**2, label='Realized Volatility (Heston)')
            ax.plot(realized_volatility_generated**2, label='Realized Volatility (Generated)')
            ax.legend()
            ax.set_title('True vs. estimated volatility (Heston) vs estimated volatility (generated)')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Volatility')
            plt.savefig(os.path.join(raw_data_outdir, 'volatility_analysis.png'))
            pdf.savefig(fig)
            plt.close(fig)


        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('VAR and CVAR Analysis')
        # VAR and CVAR
        real_var, real_cvar = compute_var_cvar(real)
        generated_var, generated_cvar = compute_var_cvar(generated_unnorm)

        real_var_mean = np.mean(real_var)
        real_cvar_mean = np.mean(real_cvar)
        generated_var_mean = np.mean(generated_var)
        generated_cvar_mean = np.mean(generated_cvar)

        sns.kdeplot(real_var, label='real', fill=True, ax=axes[0])
        sns.kdeplot(generated_var, label='generated', fill=True, ax=axes[0])
        axes[0].axvline(real_var_mean, color='blue', linestyle='--', label=f'real var mean ({real_var_mean:.3f})')
        axes[0].axvline(generated_var_mean, color='orange', linestyle='--', label=f'generated var mean ({generated_var_mean:.3f})')
        axes[0].legend()
        axes[0].set_title("KDE of VAR")

        sns.kdeplot(real_cvar, label='real', fill=True, ax=axes[1])
        sns.kdeplot(generated_cvar, label='generated', fill=True, ax=axes[1])
        axes[1].axvline(real_cvar_mean, color='blue', linestyle='--', label=f'real cvar mean ({real_cvar_mean:.3f})')
        axes[1].axvline(generated_cvar_mean, color='orange', linestyle='--', label=f'generated cvar mean ({generated_cvar_mean:.3f})')
        axes[1].legend()
        axes[1].set_title("KDE of CVAR")
        pdf.savefig(fig)
        plt.savefig(os.path.join(raw_data_outdir, 'var_cvar.png'))
        plt.close(fig)


        # Price options
        if stochastic_model != 'MarketData':
            generated_gbm_reshaped = generated_gbm.reshape(-1, 100000, n_steps)
            results = []
            for i, pricer in enumerate([price_option_call, price_option_put]):
                for K in range(70, 131, 10):
                    estimated_price_real, std_err_real, _ = pricer(paths=real_gbm.T, K=K, r=mu, T=1.0, M=n_paths)

                    estimated_prices_gen_batch = []
                    for batch in generated_gbm_reshaped:
                        estimated_price_gen, _, _ = pricer(paths=batch.T, K=K, r=mu, T=1.0, M=n_paths)
                        estimated_prices_gen_batch.append(estimated_price_gen)
                    estimated_prices_gen_avg = np.array(estimated_prices_gen_batch).mean()
                    estimated_prices_gen_std_err = np.array(estimated_prices_gen_batch).std() / np.sqrt(
                        len(estimated_prices_gen_batch))

                    if stochastic_model == 'GBM':
                        theoretical_price = price_option_bs(S0=100, K=K, r=mu, sigma=sigma, T=1.0, option_type='call' if i == 0 else 'put')
                    elif stochastic_model == 'Heston':
                        theoretical_price = heston_price(S0=100, K=K, r=mu, T=1.0, v0=v0, kappa=kwargs.get('kappa'), theta=theta,
                                                         sigma=kwargs.get('sigma_v'), rho=kwargs.get('rho'), option_type='call' if i == 0 else 'put')
                    elif stochastic_model == 'MJD':
                        theoretical_price = 0.0
                    results.append({
                        "Option Type": "Call" if i == 0 else "Put",
                        "Strike Price": K,
                        "Monte Carlo Price (Real)": round(estimated_price_real, 3),
                        "Generated Price": round(estimated_prices_gen_avg, 3),
                        "Theoretical Price": round(theoretical_price, 3),
                        "Relative Error (%)": round(100 * (estimated_price_gen - estimated_price_real) / estimated_price_real, 3),
                        "Std Error (Real)": std_err_real,
                        "Std Error (Generated)": estimated_prices_gen_std_err,
                    })

            fig, axes = plt.subplots(figsize=(10, 6))
            fig.suptitle('Option Pricing Results')
            df = pd.DataFrame(results).round(3)
            table = pd.plotting.table(axes, df, loc='center', colWidths=[0.15]*len(df.columns))
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            axes.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            df.to_csv(os.path.join(raw_data_outdir, 'option_pricing_results.csv'), index=False)

        fig, axes = plt.subplots()
        fig.suptitle('Parameters')
        params_str = str(dataset) + "\n" + f"K={K}\n" + f"Training set size: ({n_training_paths}, {n_steps-1})\n\n"
        axes.text(0.5, 0.5, params_str, fontsize=12, ha='center', va='center', bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        axes.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


def plot_distribution(axes, generated_unnorm, real):
    ks_statistic, p_value = scipy.stats.ks_2samp(real.flatten(), generated_unnorm.flatten())
    kl = scipy.special.kl_div(real.flatten(), generated_unnorm.flatten()).sum()
    wd = scipy.stats.wasserstein_distance(real.flatten(), generated_unnorm.flatten())
    # Plot KDE of Log Returns
    sns.kdeplot(real.flatten(), label='real', fill=True, ax=axes[0])
    sns.kdeplot(generated_unnorm.flatten(), label='generated', fill=True, ax=axes[0])
    axes[0].legend()
    axes[0].set_title("KDE of Log Returns")
    test_results = f'K-S Statistic: {ks_statistic:.5f}\nP-value: {p_value:.5f}'
    test_results += f'\nKL Divergence: {kl:.5f}'
    test_results += f'\nWasserstein Distance: {wd:.5f}'
    axes[1].text(0.5, 0.5, test_results,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
    axes[1].set_title('Kolmogorov-Smirnov Test Results')
    axes[1].axis('off')


def plot_sample_statistics(axes, generated_unnorm, real):
    # Calculate log return statistics
    real_mean = np.mean(real, axis=1)
    real_std = np.std(real, axis=1)
    real_skew = skew(real, axis=1)
    real_kurtosis = kurtosis(real, axis=1)
    generated_mean = np.mean(generated_unnorm, axis=1)
    generated_std = np.std(generated_unnorm, axis=1)
    generated_skew = skew(generated_unnorm, axis=1)
    generated_kurtosis = kurtosis(generated_unnorm, axis=1)
    # Plot KDE of statistics
    sns.kdeplot(real_mean, label='real', fill=True, ax=axes[0, 0])
    sns.kdeplot(generated_mean, label='generated', fill=True, ax=axes[0, 0])
    axes[0, 0].legend()
    axes[0, 0].set_title("KDE of Mean Log Returns")
    sns.kdeplot(real_std, label='real', fill=True, ax=axes[0, 1])
    sns.kdeplot(generated_std, label='generated', fill=True, ax=axes[0, 1])
    axes[0, 1].legend()
    axes[0, 1].set_title("KDE of Std Dev Log Returns")
    sns.kdeplot(real_skew, label='real', fill=True, ax=axes[1, 0])
    sns.kdeplot(generated_skew, label='generated', fill=True, ax=axes[1, 0])
    axes[1, 0].legend()
    axes[1, 0].set_title("KDE of Skewness Log Returns")
    sns.kdeplot(real_kurtosis, label='real', fill=True, ax=axes[1, 1])
    sns.kdeplot(generated_kurtosis, label='generated', fill=True, ax=axes[1, 1])
    axes[1, 1].legend()
    axes[1, 1].set_title("KDE of Kurtosis Log Returns")
    real_stats = {
        "Mean": real_mean.mean(),
        "Std": real_std.mean(),
        "Skew": real_skew.mean(),
        "Kurtosis": real_kurtosis.mean()
    }
    generated_stats = {
        "Mean": generated_mean.mean(),
        "Std": generated_std.mean(),
        "Skew": generated_skew.mean(),
        "Kurtosis": generated_kurtosis.mean()
    }
    stats_df = pd.DataFrame([real_stats, generated_stats], index=['Real', 'Generated']).round(5)
    return stats_df


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
        mu=dataset_args.get('mu', None),
        sigma=dataset_args.get('sigma_gbm', None),
        theta=dataset_args.get('theta', None),
        v0=dataset_args.get('v0', None),
        rho=dataset_args.get('rho', None),
        sigma_v=dataset_args.get('sigma_v', None),
        kappa=dataset_args.get('kappa', None),
        n_steps=dataset_args['n_steps'],
        return_log_returns=dataset_args['return_log_returns'],
        n_training_paths=dataset_args['n_paths'],
        noise_sigma=dataset_args.get('sigma', 0.0),
        corruption_probability=dataset_args.get('corruption_probability', 0.0),
        noise_type=dataset_args.get('noise_type', None),
        symbol=dataset_args.get('symbol', None),
        mu_j=dataset_args.get('mu_j', None),
        sigma_j=dataset_args.get('sigma_j', None),
        lamb=dataset_args.get('lamb', None),
    )


