import numpy as np
from scipy.optimize import minimize

from src.calibration.objective import sabr_smile_mse_loss
from src.models.sabr import SABRModel
from src.calibration.objective import heston_surface_mse_loss
from src.models.heston import HestonModel
from src.models.hull_white import HullWhiteModel
from src.calibration.objective import hull_white_swaption_surface_mse_loss

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
        2.0,
        0.04,
        0.4,
        -0.5,
        0.04,
    ),
) -> HestonModel:
    bounds = [
        (0.01, 8.0),       # kappa
        (0.001, 0.5),      # theta
        (0.01, 3.0),       # sigma
        (-0.95, 0.1),      # rho
        (0.001, 0.5),      # v0
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
        method="Nelder-Mead",
        options={
            "maxiter": 300,
            "xatol": 1e-4,
            "fatol": 1e-6,
            "disp": True,
        },
    )

    # clip parameters manually because Nelder-Mead has no hard bounds
    clipped_params = np.array([
        np.clip(result.x[0], bounds[0][0], bounds[0][1]),
        np.clip(result.x[1], bounds[1][0], bounds[1][1]),
        np.clip(result.x[2], bounds[2][0], bounds[2][1]),
        np.clip(result.x[3], bounds[3][0], bounds[3][1]),
        np.clip(result.x[4], bounds[4][0], bounds[4][1]),
    ])

    return HestonModel(
        kappa=float(clipped_params[0]),
        theta=float(clipped_params[1]),
        sigma=float(clipped_params[2]),
        rho=float(clipped_params[3]),
        v0=float(clipped_params[4]),
    )


def calibrate_hull_white_swaption_surface(
    option_maturities: np.ndarray,
    swap_maturities: np.ndarray,
    market_prices: np.ndarray,
    strike: float,
    rate: float,
    payment_frequency: int = 1,
    initial_guess: tuple[float, float] = (0.03, 0.008),
) -> HullWhiteModel:
    """
    Calibrate Hull-White parameters to a synthetic payer swaption price surface.
    """

    bounds = [
        (1e-4, 1.0),   # mean_reversion
        (1e-5, 0.20),  # volatility
    ]

    result = minimize(
        hull_white_swaption_surface_mse_loss,
        x0=np.array(initial_guess, dtype=float),
        args=(
            option_maturities,
            swap_maturities,
            market_prices,
            strike,
            rate,
            payment_frequency,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 1000,
            "ftol": 1e-14,
        },
    )

    if not result.success:
        raise RuntimeError(f"Hull-White calibration failed: {result.message}")

    mean_reversion, volatility = result.x

    return HullWhiteModel(
        mean_reversion=float(mean_reversion),
        volatility=float(volatility),
    )