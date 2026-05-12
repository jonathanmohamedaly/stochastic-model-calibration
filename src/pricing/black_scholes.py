from math import exp, log, sqrt
from typing import Literal, cast

from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

def Black_Scholes_Price(
        spot : float,
        strike : float,
        maturity : float,
        rate : float,
        volatility : float,
        option_type : OptionType = "call",
        dividend_yield : float = 0.0,
) -> float :
    """
    Black-Scholes price for European call/put options.

    Parameters
    ----------
    spot : float
        Current underlying price S0.
    strike : float
        Strike price K.
    maturity : float
        Time to maturity in years.
    rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Implied volatility.
    option_type : {"call", "put"}
        Option type.
    dividend_yield : float
        Continuous dividend yield.

    Returns
    -------
    float
        Black-Scholes option price.
    """
    if spot <= 0 : 
        raise ValueError("spot must be positive.")
    if strike <= 0 : 
        raise ValueError("strike must be positive.")
    if volatility < 0 : 
        raise ValueError("volatility must be positive.")
    if maturity <= 0 : 
        raise ValueError("maturity must be positive.")
    
    df_r = exp(-rate * maturity)
    df_q = exp(-dividend_yield * maturity)
    
    if maturity == 0 or volatility == 0 :
        forward_intrinsic_call = spot *df_q - strike * df_r

        if option_type == "call" :
            return max(forward_intrinsic_call, 0.0)
        if option_type == "put" :
            return max(-forward_intrinsic_call, 0.0)

        raise ValueError("It is either a 'call' or a 'put'")
    
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * volatility *volatility) * maturity)/(volatility * sqrt(maturity))
    d2 = d1 - volatility * maturity

    if option_type == "call" :
        return spot * df_q * float(norm.cdf(d1)) - strike * df_r * float(norm.cdf(d2))
    if option_type == "put" :
        return -spot * df_q * float(norm.cdf(-d1)) + strike * df_r * float(norm.cdf(-d2)) 

    raise ValueError("It is either a 'call' or a 'put'")  

def Black_Scholes_Vega(
    spot : float,
    strike : float,
    maturity : float,
    rate : float,
    volatility : float,
    dividend_yield : float = 0.0,
) -> float:
    """Black-Scholes Vega : derivatives of the option price with respect to the volatility"""

    if spot <= 0 : 
        raise ValueError("spot must be positive.")
    if strike <= 0 : 
        raise ValueError("strike must be positive.")
    if volatility <= 0 : 
        return 0.0
    if maturity <= 0 : 
        return 0.0
    
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * volatility *volatility) * maturity)/(volatility * sqrt(maturity))
    return spot * exp(-dividend_yield * maturity) * float(norm.pdf(d1)) * sqrt(maturity)

def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    initial_guess: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Computes Black-Scholes implied volatility using Newton-Raphson method.
    """

    if market_price <= 0:
        raise ValueError("market_price must be positive.")
    if maturity <= 0:
        raise ValueError("maturity must be positive for implied volatility.")
    if initial_guess <= 0:
        raise ValueError("initial_guess must be positive.")

    intrinsic = Black_Scholes_Price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.0,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    if option_type == "call":
        upper_bound = spot * exp(-dividend_yield * maturity)
    elif option_type == "put":
        upper_bound = strike * exp(-rate * maturity)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    if market_price < intrinsic:
        raise ValueError(
            f"market_price={market_price:.6f} is below intrinsic value={intrinsic:.6f}."
        )

    if market_price > upper_bound:
        raise ValueError(
            f"market_price={market_price:.6f} is above theoretical upper bound={upper_bound:.6f}."
        )

    sigma = initial_guess

    for _ in range(max_iterations):
        price = Black_Scholes_Price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )

        vega = Black_Scholes_Vega(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            dividend_yield=dividend_yield,
        )

        price_error = price - market_price

        if abs(price_error) < tolerance:
            return sigma

        if vega < 1e-12:
            raise RuntimeError(
                "Newton method failed because Vega is too close to zero."
            )

        sigma = sigma - price_error / vega

        if sigma <= 0:
            sigma = tolerance

        if sigma > 5.0:
            sigma = 5.0

    raise RuntimeError(
        "Implied volatility solver did not converge with Newton method."
    )

def implied_volatility_brent(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    vol_lower: float = 1e-8,
    vol_upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Computes Black-Scholes implied volatility using Brent's root-finding method.

    The solver searches for sigma such that:

        BS_price(sigma) - market_price = 0

    Returns
    -------
    float
        Implied volatility.
    """

    if market_price <= 0:
        raise ValueError("market_price must be positive.")

    if maturity <= 0:
        raise ValueError("maturity must be positive for implied volatility.")

    intrinsic = Black_Scholes_Price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.0,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    max_price = spot * exp(-dividend_yield * maturity)

    if option_type == "put":
        max_price = strike * exp(-rate * maturity)

    if market_price < intrinsic:
        raise ValueError(
            f"market_price={market_price:.6f} is below intrinsic value={intrinsic:.6f}."
        )

    if market_price > max_price:
        raise ValueError(
            f"market_price={market_price:.6f} is above theoretical upper bound={max_price:.6f}."
        )

    def objective(vol: float) -> float:
        return (
            Black_Scholes_Price(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                volatility=vol,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )
            - market_price
        )

    try:
        root = brentq(
            objective,
            vol_lower,
            vol_upper,
            xtol=tolerance,
            maxiter=max_iterations,
            full_output=False,
        )

        return float(cast(float, root))

    except ValueError as exc:
        raise RuntimeError(
            "Implied volatility solver failed. "
            "Try increasing vol_upper or check market_price consistency."
        ) from exc