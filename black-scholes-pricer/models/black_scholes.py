"""
Black-Scholes option pricing model with full Greeks calculation.
Supports European call and put options.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm


@dataclass
class OptionResult:
    price: float
    delta: float
    gamma: float
    theta: float    # per calendar day
    vega: float     # per 1% move in vol
    rho: float      # per 1% move in rate
    d1: float
    d2: float
    intrinsic_value: float
    time_value: float


def price_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> OptionResult:
    """
    Price a European option using the Black-Scholes formula.

    Args:
        S:           Spot price
        K:           Strike price
        T:           Time to expiry in years
        r:           Risk-free rate (decimal, e.g. 0.05 = 5%)
        sigma:       Volatility (decimal, e.g. 0.20 = 20%)
        option_type: 'call' or 'put'

    Returns:
        OptionResult with price, Greeks, and decomposition
    """
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    # At-expiry: pure intrinsic
    if T <= 0.0:
        return OptionResult(
            price=intrinsic,
            delta=1.0 if (option_type == "call" and S > K) else (
                -1.0 if (option_type == "put" and S < K) else 0.0
            ),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            d1=0.0, d2=0.0,
            intrinsic_value=intrinsic, time_value=0.0,
        )

    # Zero-vol edge case
    if sigma <= 0.0:
        if option_type == "call":
            price = max(S - K * np.exp(-r * T), 0.0)
        else:
            price = max(K * np.exp(-r * T) - S, 0.0)
        return OptionResult(
            price=float(price),
            delta=1.0 if price > 0 else 0.0,
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            d1=0.0, d2=0.0,
            intrinsic_value=float(intrinsic), time_value=float(price - intrinsic),
        )

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc = np.exp(-r * T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        delta = float(norm.cdf(d1))
        theta = (
            -S * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
            - r * K * disc * norm.cdf(d2)
        ) / 365.0
        rho = K * T * disc * norm.cdf(d2) / 100.0
    else:
        price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = float(norm.cdf(d1) - 1.0)
        theta = (
            -S * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
        ) / 365.0
        rho = -K * T * disc * norm.cdf(-d2) / 100.0

    gamma = float(norm.pdf(d1) / (S * sigma * sqrt_T))
    vega = float(S * norm.pdf(d1) * sqrt_T / 100.0)
    price = float(price)

    return OptionResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=float(theta),
        vega=vega,
        rho=float(rho),
        d1=float(d1),
        d2=float(d2),
        intrinsic_value=float(intrinsic),
        time_value=float(price - intrinsic),
    )


def payoff_at_expiry(
    S_range: np.ndarray,
    K: float,
    option_type: Literal["call", "put"],
) -> np.ndarray:
    """Intrinsic / payoff-at-expiry profile across a range of spot prices."""
    if option_type == "call":
        return np.maximum(S_range - K, 0.0)
    return np.maximum(K - S_range, 0.0)
