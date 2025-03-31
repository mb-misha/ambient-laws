import numpy as np
from math import factorial
from scipy.stats import norm

def merton_jump_diffusion_price(
    S0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    lamb: float,
    mu_j: float,
    sigma_j: float,
    option_type: str = 'call'
) -> float:
    """
    Closed-form solution for Merton Jump Diffusion option pricing.

    Args:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        T (float): Time to maturity
        sigma (float): Volatility of the underlying asset
        lamb (float): Jump intensity (average number of jumps per year)
        mu_j (float): Mean of jump size
        sigma_j (float): Volatility of jump size
        option_type (str, optional): 'call' or 'put', defaults to 'call'

    Returns:
        float: Theoretical option price
    """
    # Compute jump correction terms
    m = lamb * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)  # Expected jump size
    lam2 = lamb * np.exp(mu_j + 0.5 * sigma_j**2)

    # Initialize option price
    tot = 0.0

    # Compute series expansion up to 18 terms (as in the original implementation)
    for i in range(18):
        # Adjusted parameters for each term in the series
        adjusted_sigma = np.sqrt(sigma**2 + (i * sigma_j**2) / T)
        adjusted_mu = r - m + i * (mu_j + 0.5 * sigma_j**2) / T

        # Black-Scholes price with adjusted parameters
        tot += (np.exp(-lam2 * T) * (lam2 * T)**i / factorial(i)) * black_scholes_price(
            S0=S0,
            K=K,
            r=adjusted_mu,
            T=T,
            sigma=adjusted_sigma,
            option_type=option_type
        )

    return tot

def black_scholes_price(
    S0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Black-Scholes option pricing formula.

    Args:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        T (float): Time to maturity
        sigma (float): Volatility of the underlying asset
        option_type (str, optional): 'call' or 'put', defaults to 'call'

    Returns:
        float: Theoretical option price
    """
    # Compute d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Compute call and put prices
    if option_type.lower() == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")


def Merton_density(x, T, mu, sig, lam, muJ, sigJ):
    tot = 0
    for k in range(20):
        tot += (
            (lam * T) ** k
            * np.exp(-((x - mu * T - k * muJ) ** 2) / (2 * (T * sig**2 + k * sigJ**2)))
            / (factorial(k) * np.sqrt(2 * np.pi * (sig**2 * T + k * sigJ**2)))
        )
    return np.exp(-lam * T) * tot

def log_likely_Merton(x, data, T):
    return (-1) * np.sum(np.log(Merton_density(data, T, x[0], x[1], x[2], x[3], x[4])))


def Merton_density_adjusted(x, T, mu, sigma, lamb, muJ, sigmaJ, K=20):
    drift_adj = mu - lamb * (muJ + 0.5 * sigmaJ ** 2)
    total = 0
    for k in range(K):
        variance = sigma ** 2 * T + k * sigmaJ ** 2
        mean = drift_adj * T + k * muJ
        weight = (lamb * T) ** k * np.exp(-((x - mean) ** 2) / (2 * variance))
        weight /= (factorial(k) * np.sqrt(2 * np.pi * variance))
        total += weight
    return np.exp(-lamb * T) * total

# --- Log-likelihood function ---
def log_likelihood_merton_adj(params, data, T):
    mu, sigma, lamb, muJ, sigmaJ = params
    # Avoid invalid regions
    if sigma <= 0 or sigmaJ <= 0 or lamb <= 0:
        return np.inf
    pdf_vals = Merton_density_adjusted(data, T, mu, sigma, lamb, muJ, sigmaJ)
    # Avoid log(0) by clipping pdf values
    return -np.sum(np.log(np.clip(pdf_vals, 1e-12, None)))




import numpy as np

def estimate_mjd_parameters(series, dt, threshold=3):
    """
    Estimate parameters of the Merton Jump Diffusion model from price series.

    Parameters:
    - series: 2D array of shape (n_paths, n_steps)
    - dt: Time step size.
    - threshold: Threshold (in standard deviations) to identify jumps.

    Returns:
    - params: Dictionary of arrays with shape (n_paths,) for each parameter.
    """
    if series.ndim == 1:
        series = series[np.newaxis, :]  # Make it (1, n_steps)

    log_returns = np.diff(np.log(series), axis=1)
    n_paths, n_returns = log_returns.shape

    mu_hat = np.mean(log_returns, axis=1, keepdims=True)
    sigma_hat = np.std(log_returns, axis=1, ddof=1, keepdims=True)

    # Identify jumps
    abs_deviation = np.abs(log_returns - mu_hat)
    is_jump = abs_deviation > threshold * sigma_hat
    is_no_jump = ~is_jump

    # Jump intensity λ
    lamb_hat = np.sum(is_jump, axis=1) / (n_returns * dt)

    # Jump stats
    jump_sizes = np.where(is_jump, log_returns, np.nan)
    mu_j_hat = np.nanmean(jump_sizes, axis=1)
    sigma_j_hat = np.nanstd(jump_sizes, axis=1, ddof=1)

    # Set to nan if insufficient jump data
    jump_counts = np.sum(is_jump, axis=1)
    mu_j_hat = np.where(jump_counts == 0, np.nan, mu_j_hat)
    sigma_j_hat = np.where(jump_counts <= 1, np.nan, sigma_j_hat)

    # Diffusion component
    diffusion_returns = np.where(is_no_jump, log_returns, np.nan)
    mu_diff_hat = np.nanmean(diffusion_returns, axis=1) / dt
    sigma_diff_hat = np.nanstd(diffusion_returns, axis=1, ddof=1) / np.sqrt(dt)

    # Adjusted drift
    k_hat = np.exp(mu_j_hat + 0.5 * sigma_j_hat**2) - 1
    mu_hat_adj = mu_diff_hat + lamb_hat * k_hat

    params = {
        "mu": mu_hat_adj,
        "sigma": sigma_diff_hat,
        "lamb": lamb_hat,
        "mu_j": mu_j_hat,
        "sigma_j": sigma_j_hat,
    }

    return params
