from math import log, sqrt

from src.models.sabr import SABRModel

def hagan_implied_volatility(
        forward : float,
        strike : float,
        maturity : float,
        model : SABRModel,
) -> float : 
    """
    Hagan SABR implied volatility approximation.

    Parameters
    ----------
    forward : float
        Forward price F.
    strike : float
        Strike price K.
    maturity : float
        Time to maturity T.
    model : SABRModel
        SABR model parameters.

    Returns
    -------
    float
        Black implied volatility.
    """

    if forward <= 0 :
        raise ValueError("forward must be positive")
    
    if strike <= 0 :
        raise ValueError("strike must be positive")
    
    if maturity <= 0 :
        raise ValueError("maturity must be positive")
    
    alpha = model.alpha
    beta = model.beta
    rho = model.rho
    nu = model.nu

    one_minus_beta = 1 - beta

    #ATM
    if abs(forward - strike) <= 1e-12 :

        fk_beta = forward ** one_minus_beta

        correction = (
            ((one_minus_beta**2) / 24.0) * (alpha**2 / (forward ** (2.0 * one_minus_beta)))
            + (rho * beta * nu * alpha) / (4.0 * (forward ** one_minus_beta))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        ) * maturity

        return (alpha / fk_beta) * (1.0 + correction)

    log_fk = log(forward / strike)
    fk = forward * strike
    fk_beta = fk ** (0.5 * one_minus_beta)

    z = (nu / alpha) * fk_beta * log_fk

    x_z = log(
        (sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho)
        / (1.0 - rho)
    )

    numerator = alpha * z

    denominator = (
        fk_beta
        * (
            1.0
            + ((one_minus_beta**2) / 24.0) * log_fk**2
            + ((one_minus_beta**4) / 1920.0) * log_fk**4
        )
        * x_z
    )

    correction = (
        ((one_minus_beta**2) / 24.0) * (alpha**2 / (fk ** one_minus_beta))
        + (rho * beta * nu * alpha) / (4.0 * fk_beta)
        + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
    ) * maturity

    return (numerator / denominator) * (1.0 + correction)