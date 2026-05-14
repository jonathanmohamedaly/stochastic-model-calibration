import numpy as np
from scipy.optimize import minimize

from src.calibration.objective import sabr_smile_mse_loss
from src.models.sabr import SABRModel
from src.calibration.objective import heston_surface_mse_loss
from src.models.heston import HestonModel

def calibrate_sabr_smile(
        strikes : np.ndarray,
        market_vols : np.ndarray,
        forward : float,
        maturity : float,
        beta : float = 0.5,
        initial_guess : tuple[float, float, float] = (0.2, -0.2, 0.5),
) -> SABRModel : 

    bounds = [
        (1e-4, 5.0),
        (-0.999, 0.999),
        (1e-4, 5.0),
    ]

    result = minimize(
        sabr_smile_mse_loss,
        x0= np.array(initial_guess),
        args= (strikes, market_vols, forward, maturity, beta),
        method="L-BFGS-B",
        bounds = bounds,
        options= {
            "maxiter" : 10_000,
            "ftol" : 1e-14,
        },
    )

    if not result.success:
        raise RuntimeError(f"SABR calibration failed : {result.message}")
    
    alpha, rho, nu = result.x

    return SABRModel(
            alpha= float(alpha),
            beta = float(beta),
            rho = float(rho),
            nu = float(nu)
        )

def calibrate_heston_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_vol_surface: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    initial_guess: tuple[float, float, float, float, float] = (
        1.5,
        0.04,
        0.3,
        -0.5,
        0.04,
    ),
) -> HestonModel:
    """
    Calibrate Heston parameters to an implied volatility surface.
    """

    bounds = [
        (1e-4, 10.0),      # kappa
        (1e-4, 1.0),       # theta
        (1e-4, 5.0),       # sigma
        (-0.999, 0.999),   # rho
        (1e-4, 1.0),       # v0
    ]

    result = minimize(
        heston_surface_mse_loss,
        x0=np.array(initial_guess, dtype=float),
        args=(
            strikes,
            maturities,
            market_vol_surface,
            spot,
            rate,
            dividend_yield,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
        },
    )

    if not result.success:
        raise RuntimeError(f"Heston calibration failed: {result.message}")

    kappa, theta, sigma, rho, v0 = result.x

    return HestonModel(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        rho=float(rho),
        v0=float(v0),
    )