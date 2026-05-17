import numpy as np
from scipy.stats import norm

from src.models.hull_white import HullWhiteModel

def discount_factor(rate : float, maturity : float) -> float :
    if maturity < 0 :
        raise ValueError("maturity must be positive")
    return float(np.exp(-rate * maturity))

def hull_white_B(
        t : float,
        T : float,
        model : HullWhiteModel
)->float :
    a = model.mean_reversion

    if T < t :
        raise ValueError("T must be greater or equal than t")
    
    return (1.0 - np.exp(-a* (T -t))) / a

def hull_white_bond_option_volatility(
        option_maturity : float,
        bond_maturity : float, 
        model : HullWhiteModel,
) -> float :
    """
    Volatility of zero-coupon bond option under one-factor Hull-White.
    """

    a = model.mean_reversion
    sigma = model.volatility

    if bond_maturity <= option_maturity :
        raise ValueError("bond_maturity must be strictly greater than option_maturity")
    
    B = hull_white_B(option_maturity, bond_maturity, model)

    variance = (
        sigma**2
        * (1.0 - np.exp(-2.0 * a * option_maturity))
        / (2.0 * a)
        * B**2
    )

    return float(np.sqrt(variance))

def zero_coupon_bond_option_price(
        option_type :str,
        strike : float,
        option_maturity : float,
        bond_maturity : float, 
        rate : float, 
        model : HullWhiteModel,
) -> float :
    """
    European option on a zero-coupon bond using Black formula.
    """

    if strike <= 0:
        raise ValueError("strike must be positive")
    
    P0S = discount_factor(rate, option_maturity)
    P0T = discount_factor(rate, bond_maturity)

    vol = hull_white_bond_option_volatility(option_maturity, bond_maturity, model)

    if vol <= 0 : 
        intrinsic = P0T - strike * P0S

        if option_type == "call" :
            return max(intrinsic, 0.0)
        
        if option_type == "put" :
            return max(-intrinsic, 0.0)
        
        else :
            raise ValueError("it is either call or put")
        
    d1 = np.log(P0T / (strike * P0S)) / vol + 0.5 * vol
    d2 = d1 - vol

    if option_type == "call":
        return float(P0T * norm.cdf(d1) - strike * P0S * norm.cdf(d2))
    
    if option_type == "put":
        return float(strike * P0S * norm.cdf(-d2) - P0T * norm.cdf(-d1))
    
    raise ValueError("it is either call or put")

def forward_swap_rate(
        rate : float, 
        option_maturity : float, 
        swap_maturity : float,
        payment_frequency : int = 1,
) -> float : 
    """
    Forward swap rate under a flat zero-coupon curve.
    """

    if swap_maturity <= option_maturity : 
        raise ValueError("swap_maturity must be greater strictly than option_maturity")
    
    if payment_frequency <= 0 : 
        raise ValueError("payment_frequency must be greater than 0")
    
    delta = 1.0 / payment_frequency

    payment_times = np.arange(
        option_maturity + delta,
        swap_maturity + 1e-12, 
        delta,
    )

    discount_factors = np.array([
        discount_factor(rate, t) for t in payment_times
    ])

    annuity = delta * np.sum(discount_factors)

    p_start = discount_factor(rate, option_maturity)
    p_end = discount_factor(rate, swap_maturity)

    return float((p_start - p_end) / annuity)

def swap_annuity(
        rate : float,
        option_maturity : float, 
        swap_maturity : float,
        payment_frequency : float = 1,
) -> float :
    """
    Present value of the fixed leg annuity.
    """

    if swap_maturity <= option_maturity : 
        raise ValueError("swap_maturity must be greater strictly than option_maturity")
    
    delta = 1 / payment_frequency

    payment_times = np.arange(
        option_maturity + delta,
        swap_maturity + 1e-12, 
        delta,
    )

    discount_factors = np.array([
        discount_factor(rate, t) for t in payment_times
    ])

    return float(delta * np.sum(discount_factors))

def hull_white_swaption_volatility(
        option_maturity : float, 
        swap_maturity : float,
        model : HullWhiteModel,
) -> float :
    """
    Simplified effective swaption volatility under Hull-White.

    This is an approximation suitable for synthetic calibration experiments.
    """

    a = model.mean_reversion
    sigma = model.volatility

    tenor = swap_maturity - option_maturity

    if tenor < 0 :
        raise ValueError("swap_maturity must be greater strictly than option_maturity")
    
    variance = (
        sigma**2
        * (1.0 - np.exp(-2.0 * a * option_maturity))
        / (2.0 * a)
        * ((1.0 - np.exp(-a * tenor)) / a) ** 2
    )

    return float(np.sqrt(variance / option_maturity))

def payer_swaption_price(
    strike: float,
    option_maturity: float,
    swap_maturity: float,
    rate: float,
    model: HullWhiteModel,
    payment_frequency: int = 1,
) -> float:
    """
    Approximate payer swaption price using Black formula.
    """

    if strike <= 0:
        raise ValueError("strike must be positive.")

    forward_rate = forward_swap_rate(
        rate=rate,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        payment_frequency=payment_frequency,
    )

    annuity = swap_annuity(
        rate=rate,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        payment_frequency=payment_frequency,
    )

    vol = hull_white_swaption_volatility(
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        model=model,
    )

    if vol <= 0:
        return annuity * max(forward_rate - strike, 0.0)

    total_vol = vol * np.sqrt(option_maturity)

    d1 = np.log(forward_rate / strike) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol

    return float(
        annuity
        * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    )


def receiver_swaption_price(
    strike: float,
    option_maturity: float,
    swap_maturity: float,
    rate: float,
    model: HullWhiteModel,
    payment_frequency: int = 1,
) -> float:
    """
    Approximate receiver swaption price using Black formula.
    """

    if strike <= 0:
        raise ValueError("strike must be positive.")

    forward_rate = forward_swap_rate(
        rate=rate,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        payment_frequency=payment_frequency,
    )

    annuity = swap_annuity(
        rate=rate,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        payment_frequency=payment_frequency,
    )

    vol = hull_white_swaption_volatility(
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        model=model,
    )

    if vol <= 0:
        return annuity * max(strike - forward_rate, 0.0)

    total_vol = vol * np.sqrt(option_maturity)

    d1 = np.log(forward_rate / strike) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol

    return float(
        annuity
        * (strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))
    )