"""
REST API routes for Black-Scholes pricing.

Endpoints:
  GET  /api/health         - liveness check
  POST /api/price          - price a single option + Greeks
  POST /api/sensitivity    - sensitivity curve for one varying parameter
"""

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.black_scholes import payoff_at_expiry, price_option

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PriceRequest(BaseModel):
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., ge=0, description="Time to expiry (years)")
    r: float = Field(..., description="Risk-free rate (decimal, e.g. 0.05)")
    sigma: float = Field(..., ge=0, description="Volatility (decimal, e.g. 0.20)")
    option_type: Literal["call", "put"] = "call"


class SensitivityRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., ge=0)
    r: float
    sigma: float = Field(..., ge=0)
    option_type: Literal["call", "put"] = "call"
    vary: Literal["S", "K", "T", "r", "sigma"] = Field(
        "S", description="Which parameter to sweep"
    )
    points: int = Field(100, ge=10, le=500, description="Number of data points")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sensible sweep ranges relative to current value
_RANGE_FACTORIES = {
    "S":     lambda p: (max(p.S * 0.40, 0.01),     p.S * 1.60),
    "K":     lambda p: (max(p.K * 0.40, 0.01),     p.K * 1.60),
    "T":     lambda p: (0.01,                        max(p.T * 2.5, 3.0)),
    "r":     lambda p: (0.00,                        0.20),
    "sigma": lambda p: (0.01,                        min(p.sigma * 3.5 + 0.05, 2.0)),
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model": "Black-Scholes"}


@router.post("/price")
async def calculate_price(req: PriceRequest):
    """Return the BS price and all Greeks for the supplied parameters."""
    try:
        res = price_option(req.S, req.K, req.T, req.r, req.sigma, req.option_type)
        return {
            "price":           round(res.price, 6),
            "delta":           round(res.delta, 6),
            "gamma":           round(res.gamma, 6),
            "theta":           round(res.theta, 6),
            "vega":            round(res.vega, 6),
            "rho":             round(res.rho, 6),
            "d1":              round(res.d1, 6),
            "d2":              round(res.d2, 6),
            "intrinsic_value": round(res.intrinsic_value, 6),
            "time_value":      round(res.time_value, 6),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/sensitivity")
async def sensitivity(req: SensitivityRequest):
    """
    Return a curve of prices/Greeks as one parameter varies.
    Also includes payoff-at-expiry values when varying spot (S).
    """
    try:
        low, high = _RANGE_FACTORIES[req.vary](req)
        xs = np.linspace(low, high, req.points)

        data = []
        for x in xs:
            params = dict(S=req.S, K=req.K, T=req.T, r=req.r, sigma=req.sigma)
            params[req.vary] = float(x)
            res = price_option(
                params["S"], params["K"], params["T"],
                params["r"], params["sigma"],
                req.option_type,
            )
            row = {
                "x":     round(float(x), 6),
                "price": round(res.price, 6),
                "delta": round(res.delta, 6),
                "gamma": round(res.gamma, 6),
                "theta": round(res.theta, 6),
                "vega":  round(res.vega, 6),
                "rho":   round(res.rho, 6),
            }
            data.append(row)

        # Payoff-at-expiry overlay is only meaningful when sweeping spot
        if req.vary == "S":
            payoffs = payoff_at_expiry(xs, req.K, req.option_type).tolist()
            for row, poff in zip(data, payoffs):
                row["payoff"] = round(poff, 6)

        current_val = getattr(req, req.vary)

        return {
            "vary":          req.vary,
            "current_value": current_val,
            "data":          data,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
