import numpy as np
from scipy.integrate import quad

from src.models.heston import HestonModel
from src.pricing.black_scholes import implied_volatility_brent

def heston_characteristic_function(
        u :complex,
        spot : float, 
        maturity : float, 
        rate : float,
        dividend_yield : float,
        model : HestonModel,
)-> complex : 
    """Heston characterisitc function of log(S_T)"""

    if spot <= 0 :
        raise ValueError("spot must be positive ")
    
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    
    kappa = model.kappa
    theta = model.theta
    sigma = model.sigma
    rho = model.rho
    v0 = model.v0

    x0 = np.log(spot)

    i = 1j

    d = np.sqrt(
        (rho * sigma * i * u - kappa) ** 2
        + sigma**2 * (i * u + u**2)
    )

    g = (kappa - rho * sigma * i * u - d) / (
        kappa - rho * sigma * i * u + d
    )

    exp_dt = np.exp(-d * maturity)

    C = (
        i * u * (x0 + (rate - dividend_yield) * maturity)
        + (kappa * theta / sigma**2)
        * (
            (kappa - rho * sigma * i * u - d) * maturity
            - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
        )
    )

    D = (
        (kappa - rho * sigma * i * u - d)
        / sigma**2
        * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    )

    return np.exp(C + D * v0)

def _heston_probability_integrand(
    u: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    model: HestonModel,
    probability_index: int,
) -> float:
    i = 1j

    if probability_index == 1:
        numerator = heston_characteristic_function(
            u - i,
            spot,
            maturity,
            rate,
            dividend_yield,
            model,
        )
        denominator = (
            i
            * u
            * heston_characteristic_function(
                -i,
                spot,
                maturity,
                rate,
                dividend_yield,
                model,
            )
        )
    elif probability_index == 2:
        numerator = heston_characteristic_function(
            u,
            spot,
            maturity,
            rate,
            dividend_yield,
            model,
        )
        denominator = i * u
    else:
        raise ValueError("probability_index must be 1 or 2.")

    value = np.exp(-i * u * np.log(strike)) * numerator / denominator

    return float(np.real(value))


def heston_probability(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    model: HestonModel,
    probability_index: int,
    integration_limit: float = 100.0,
) -> float:
    integral, _ = quad(
        _heston_probability_integrand,
        1e-8,
        integration_limit,
        args=(
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            model,
            probability_index,
        ),
        limit=500,
    )

    return 0.5 + integral / np.pi


def heston_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    model: HestonModel,
    dividend_yield: float = 0.0,
    integration_limit: float = 100.0,
) -> float:
    """
    European call price under the Heston model using Fourier integration.
    """

    p1 = heston_probability(
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        model,
        probability_index=1,
        integration_limit=integration_limit,
    )

    p2 = heston_probability(
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        model,
        probability_index=2,
        integration_limit=integration_limit,
    )

    return (
        spot * np.exp(-dividend_yield * maturity) * p1
        - strike * np.exp(-rate * maturity) * p2
    )


def heston_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    model: HestonModel,
    dividend_yield: float = 0.0,
    integration_limit: float = 100.0,
) -> float:
    """
    European put price under the Heston model using put-call parity.
    """

    call = heston_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        model=model,
        dividend_yield=dividend_yield,
        integration_limit=integration_limit,
    )

    return (
        call
        - spot * np.exp(-dividend_yield * maturity)
        + strike * np.exp(-rate * maturity)
    )

def heston_implied_volatility(
        spot : float,
        strike : float,
        maturity : float,
        rate : float,
        model : HestonModel,
        option_type : str = "call",
        dividend_yield : float = 0.0,
        integration_limit : float = 100.0,
) -> float :
    """ Convert Heston option price into BS implied volatility"""

    if option_type == "call":
        price = heston_call_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            model=model,
            dividend_yield=dividend_yield,
            integration_limit=integration_limit,
        )

    elif option_type == "put":
        price = heston_put_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            model=model,
            dividend_yield=dividend_yield,
            integration_limit=integration_limit,
        )

    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return implied_volatility_brent(
        market_price=price,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )