import numpy as np

from src.models.sabr import SABRModel
from src.pricing.sabr_formula import hagan_implied_volatility

def sabr_smile_mse_loss(
        parameters : np.ndarray,
        strikes : np.ndarray,
        markets_vol : np.ndarray,
        forward : float,
        maturity : float,
        beta : float,
) -> float :
    """
    Mean squared error between market implied vols and SABR implied vols.

    parameters = [alpha, rho, nu]
    beta is fixed during calibration.
    """
    alpha, rho, nu = parameters

    try:
        model = SABRModel(
            alpha= float(alpha),
            beta = float(beta),
            rho = float(rho),
            nu = float(nu)
        )

        model_vols = np.array(
            [
                hagan_implied_volatility(
                    forward= forward,
                    strike= float(strike),
                    maturity= maturity, 
                    model= model
                )
                for strike in strikes
            ]
        )
        return float(np.mean((model_vols - markets_vol) ** 2))
    
    except (ValueError, FloatingPointError, OverflowError):
        return 1e10