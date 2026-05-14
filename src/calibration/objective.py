import numpy as np

from src.models.sabr import SABRModel
from src.pricing.sabr_formula import hagan_implied_volatility
from src.models.heston import HestonModel
from src.pricing.heston_pricer import heston_implied_volatility

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
    
def heston_surface_mse_loss(
        parameters : np.ndarray,
        strikes : np.ndarray,
        maturities : np.ndarray,
        market_vol_surface : np.ndarray,
        spot : float,
        rate : float,
        dividend_yield : float=0.0,
)-> float :
    """
    MSE loss between market implied volatility surface and Heston implied volatility surface.

    parameters = [kappa, theta, sigma, rho, v0]
    """

    kappa, theta, sigma, rho, v0 = parameters

    try : 
        model = HestonModel(
            kappa= float(kappa),
            theta= float(theta),
            sigma= float(sigma),
            rho= float(rho),
            v0= float(v0),
        )

        model_vol_surface = np.zeros_like(market_vol_surface, dtype=float)

        for i, maturity in enumerate(maturities):
            for j, strike in enumerate(strikes):
                model_vol_surface[i, j] = heston_implied_volatility(
                    spot= spot, 
                    strike= float(strike),
                    maturity= float(maturity),
                    rate= rate,
                    model= model, 
                    dividend_yield= dividend_yield,
                )
        
        error = model_vol_surface - market_vol_surface
        return float(np.mean(error**2))
    
    except:
        return 1e10