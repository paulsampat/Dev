"""
Black-Scholes MCP Server
========================
Exposes the local pricing model to Claude (and any MCP-compatible host)
via the Model Context Protocol.  No HTTP dependency — imports the core
pricing logic directly from models.black_scholes.

Run with:
    python mcp_server.py               # stdio transport (Claude Desktop)
    mcp dev mcp_server.py              # browser-based MCP Inspector (testing)

Register with Claude Desktop by adding to
~/Library/Application Support/Claude/claude_desktop_config.json:

    {
      "mcpServers": {
        "black-scholes-pricer": {
          "command": "/opt/anaconda3/envs/anaconda-ml-ai/bin/python",
          "args": ["/Users/paulsampat/Dev/black-scholes-pricer/mcp_server.py"],
          "env": { "PYTHONPATH": "/Users/paulsampat/Dev/black-scholes-pricer" }
        }
      }
    }

IMPORTANT: Do NOT add any print() calls here — the MCP stdio transport uses
stdout for JSON-RPC; any stray output will corrupt the wire protocol.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np

from mcp.server.fastmcp import FastMCP

# Alias to free up the plain names for the MCP tool functions below
from models.black_scholes import payoff_at_expiry as _payoff_at_expiry
from models.black_scholes import price_option as _price_option

# ── Server instance ────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="black-scholes-pricer",
    instructions=(
        "Price European call and put options using the Black-Scholes formula. "
        "All rate/volatility inputs are decimals (e.g. r=0.05 = 5%, sigma=0.20 = 20%). "
        "Theta is per calendar day. Vega and Rho are each per 1% move in their input."
    ),
)

# ── Sweep-range factories ──────────────────────────────────────────────────────
# Re-declared here (not imported from routers/pricing.py) to keep this file
# free of FastAPI / Pydantic dependencies.
_RANGE_FACTORIES: dict[str, object] = {
    "S":     lambda S, K, T, r, sigma: (max(S * 0.40, 0.01),     S * 1.60),
    "K":     lambda S, K, T, r, sigma: (max(K * 0.40, 0.01),     K * 1.60),
    "T":     lambda S, K, T, r, sigma: (0.01,                     max(T * 2.5, 3.0)),
    "r":     lambda S, K, T, r, sigma: (0.00,                     0.20),
    "sigma": lambda S, K, T, r, sigma: (0.01,                     min(sigma * 3.5 + 0.05, 2.0)),
}


# ── Tool: price_option ────────────────────────────────────────────────────────
@mcp.tool()
def price_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> str:
    """
    Price a single European option using Black-Scholes and return the price
    together with all five Greeks and a price decomposition.

    Args:
        S:           Current spot (underlying) price. Must be > 0.
        K:           Strike price. Must be > 0.
        T:           Time to expiry in years (e.g. 0.25 = 3 months). Must be >= 0.
        r:           Continuously-compounded risk-free rate as a decimal
                     (e.g. 0.05 = 5 %).
        sigma:       Annualised implied/realised volatility as a decimal
                     (e.g. 0.20 = 20 %). Must be >= 0.
        option_type: "call" (long the upside) or "put" (long the downside).
                     Defaults to "call".

    Returns:
        JSON string with the following structure:
        {
          "option_type": "call" | "put",
          "inputs":      { S, K, T, r, sigma },
          "price":       float,
          "greeks": {
            "delta":         float,   # dV/dS
            "gamma":         float,   # d²V/dS²
            "theta_per_day": float,   # dV/dt per calendar day (usually negative)
            "vega_per_1pct": float,   # dV/d_sigma per 1 % vol move
            "rho_per_1pct":  float    # dV/dr per 1 % rate move
          },
          "decomposition": {
            "intrinsic_value": float,
            "time_value":      float
          },
          "intermediates": { "d1": float, "d2": float }
        }
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be > 0, got {K}")
    if T < 0:
        raise ValueError(f"Time to expiry T must be >= 0, got {T}")
    if sigma < 0:
        raise ValueError(f"Volatility sigma must be >= 0, got {sigma}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    res = _price_option(S, K, T, r, sigma, option_type)

    payload = {
        "option_type": option_type,
        "inputs": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma},
        "price": round(res.price, 6),
        "greeks": {
            "delta":         round(res.delta, 6),
            "gamma":         round(res.gamma, 6),
            "theta_per_day": round(res.theta, 6),
            "vega_per_1pct": round(res.vega, 6),
            "rho_per_1pct":  round(res.rho, 6),
        },
        "decomposition": {
            "intrinsic_value": round(res.intrinsic_value, 6),
            "time_value":      round(res.time_value, 6),
        },
        "intermediates": {
            "d1": round(res.d1, 6),
            "d2": round(res.d2, 6),
        },
    }
    return json.dumps(payload, indent=2)


# ── Tool: sensitivity_analysis ────────────────────────────────────────────────
@mcp.tool()
def sensitivity_analysis(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    vary: Literal["S", "K", "T", "r", "sigma"] = "S",
    points: int = 20,
) -> str:
    """
    Sweep one input parameter across its natural range and return a compact
    sensitivity curve showing how the option price and Greeks respond.

    Args:
        S:           Base spot price (used as reference when vary != "S").
        K:           Strike price.
        T:           Time to expiry in years.
        r:           Risk-free rate (decimal).
        sigma:       Volatility (decimal).
        option_type: "call" or "put". Defaults to "call".
        vary:        Which parameter to sweep: "S", "K", "T", "r", or "sigma".
                     Defaults to "S" (spot price).
        points:      Number of evenly-spaced data points to return.
                     Capped at 20 internally to keep the response concise.

    Returns:
        JSON string:
        {
          "vary":          "S" | "K" | "T" | "r" | "sigma",
          "current_value": float,
          "range":         { "low": float, "high": float },
          "option_type":   "call" | "put",
          "curve": [
            { "x": float, "price": float, "delta": float, "gamma": float,
              "theta": float, "vega": float, "rho": float },
            ...
          ],
          "payoff": [float, ...]   # only present when vary="S"
        }
    """
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if vary not in _RANGE_FACTORIES:
        raise ValueError(f"vary must be one of {list(_RANGE_FACTORIES)}, got {vary!r}")

    # Cap to 20 to keep JSON response compact for the AI context window
    n = min(max(int(points), 2), 20)

    low, high = _RANGE_FACTORIES[vary](S, K, T, r, sigma)
    xs = np.linspace(low, high, n)

    base = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
    current_value = base[vary]

    curve = []
    for x in xs:
        params = dict(base)
        params[vary] = float(x)
        res = _price_option(
            params["S"], params["K"], params["T"],
            params["r"], params["sigma"],
            option_type,
        )
        curve.append({
            "x":     round(float(x), 6),
            "price": round(res.price, 6),
            "delta": round(res.delta, 6),
            "gamma": round(res.gamma, 6),
            "theta": round(res.theta, 6),
            "vega":  round(res.vega, 6),
            "rho":   round(res.rho, 6),
        })

    payload: dict = {
        "vary":          vary,
        "current_value": current_value,
        "range":         {"low": round(low, 6), "high": round(high, 6)},
        "option_type":   option_type,
        "curve":         curve,
    }

    # Payoff-at-expiry overlay is only meaningful when sweeping the spot price
    if vary == "S":
        payoffs = _payoff_at_expiry(xs, K, option_type).tolist()
        payload["payoff"] = [round(p, 6) for p in payoffs]

    return json.dumps(payload, indent=2)


# ── Resource: Black-Scholes formula reference ─────────────────────────────────
@mcp.resource(
    uri="bs://formula",
    name="Black-Scholes Formula Reference",
    description="Full Black-Scholes formula, Greeks definitions, and model assumptions.",
    mime_type="text/plain",
)
def bs_formula_resource() -> str:
    """Static reference text for the Black-Scholes model."""
    return """\
BLACK-SCHOLES EUROPEAN OPTION PRICING FORMULA
=============================================

Variables
---------
  S     = current spot (underlying) price
  K     = option strike price
  T     = time to expiry, in years
  r     = continuously-compounded risk-free interest rate (decimal)
  sigma = annualised volatility (decimal)
  N(x)  = cumulative standard normal CDF
  n(x)  = standard normal PDF

Intermediate values
-------------------
  d1 = [ ln(S/K) + (r + sigma²/2) * T ] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)

Option prices
-------------
  Call  C = S * N(d1)  -  K * e^(-rT) * N(d2)
  Put   P = K * e^(-rT) * N(-d2)  -  S * N(-d1)

Put-call parity
---------------
  C - P = S - K * e^(-rT)

Greeks
------
  Delta   dV/dS
      Call:  N(d1)          Put:  N(d1) - 1

  Gamma   d²V/dS²  (identical for call and put)
      n(d1) / (S * sigma * sqrt(T))

  Theta   dV/dt  per calendar day
      Call: [ -S*n(d1)*sigma/(2*sqrt(T)) - r*K*e^(-rT)*N(d2)  ] / 365
      Put:  [ -S*n(d1)*sigma/(2*sqrt(T)) + r*K*e^(-rT)*N(-d2) ] / 365

  Vega    dV/d_sigma  per 1% move in volatility
      S * n(d1) * sqrt(T) / 100

  Rho     dV/dr  per 1% move in rate
      Call:  K * T * e^(-rT) * N(d2)  / 100
      Put:  -K * T * e^(-rT) * N(-d2) / 100

Price decomposition
-------------------
  Intrinsic value = max(S-K, 0) for call  |  max(K-S, 0) for put
  Time value      = option price - intrinsic value

Model assumptions
-----------------
  European exercise only (no early exercise).
  Constant volatility and risk-free rate over [0, T].
  Underlying pays no dividends.
  Continuous frictionless trading.
"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Default transport is stdio — Claude Desktop communicates over stdin/stdout.
    mcp.run()
