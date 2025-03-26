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