import numpy as np

from src.models.sabr import SABRModel
from src.pricing.sabr_formula import hagan_implied_volatility
from src.models.heston import HestonModel
from src.pricing.heston_pricer import heston_implied_volatility
from src.models.hull_white import HullWhiteModel
from src.pricing.swaption_pricer import payer_swaption_price

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

def hull_white_swaption_surface_mse_loss(
    parameters: np.ndarray,
    option_maturities: np.ndarray,
    swap_maturities: np.ndarray,
    market_prices: np.ndarray,
    strike: float,
    rate: float,
    payment_frequency: int = 1,
) -> float:
    """
    MSE loss between market swaption prices and Hull-White model prices.

    parameters = [mean_reversion, volatility]
    """

    mean_reversion, volatility = parameters

    try:
        model = HullWhiteModel(
            mean_reversion=float(mean_reversion),
            volatility=float(volatility),
        )

        model_prices = np.zeros_like(market_prices, dtype=float)

        for i, option_maturity in enumerate(option_maturities):
            for j, swap_maturity in enumerate(swap_maturities):
                if np.isnan(market_prices[i, j]):
                    model_prices[i, j] = np.nan
                    continue

                model_prices[i, j] = payer_swaption_price(
                    strike=strike,
                    option_maturity=float(option_maturity),
                    swap_maturity=float(swap_maturity),
                    rate=rate,
                    model=model,
                    payment_frequency=payment_frequency,
                )

        errors = model_prices - market_prices

        return float(np.nanmean(errors**2))

    except Exception:
        return 1e10