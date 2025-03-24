import argparse
import json
import os

from dataset_gbm import GBMGenerativeDataset, estimate_parameters, reverse_log_return, HestonGenerativeDataset
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
from heston_closed_form_solution import heston_price

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


def process_data(generated_filepath, normalization, stochastic_model, mu, sigma, theta, v0, n_steps, return_log_returns,
                 n_training_paths, noise_sigma, n_paths=100000, K=110, **kwargs):
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
    else:
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


    generated_gbm = generated_gbm.reshape(10, 100000, n_steps)
    # Price option
    results = []
    for K in range(70, 131, 10):
        estimated_price_real, std_err_real, _ = price_option(paths=real_gbm.T, K=K, r=mu, T=1.0, M=n_paths)


        estimated_prices_gen_batch = []
        for batch in generated_gbm:
            estimated_price_gen, _, _ = price_option(paths=batch.T, K=K, r=mu, T=1.0, M=n_paths)
            estimated_prices_gen_batch.append(estimated_price_gen)
        estimated_prices_gen_avg = np.array(estimated_prices_gen_batch).mean()
        estimated_prices_gen_std_err = np.array(estimated_prices_gen_batch).std() / np.sqrt(len(estimated_prices_gen_batch))


        if stochastic_model == 'GBM':
            theoretical_price = price_option_bs(S0=100, K=K, r=mu, sigma=sigma, T=1.0)
        else:
            theoretical_price = heston_price(S0=100, K=K, r=mu, T=1.0, v0=v0, kappa=kwargs.get('kappa'), theta=theta, sigma=kwargs.get('sigma_v'), rho=kwargs.get('rho'))
        results.append({
            "Strike Price": K,
            "Monte Carlo Price (Real)": round(estimated_price_real, 3),
            "Generated Price": round(estimated_prices_gen_avg, 3),
            "Theoretical Price": round(theoretical_price, 3),
            "Relative Error (%)": round(100 * (estimated_price_gen - estimated_price_real) / estimated_price_real, 3),
            "Std Error (Real)": std_err_real,
            "Std Error (Generated)": estimated_prices_gen_std_err,
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "option_pricing_results.csv"), index=False)



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
    )

