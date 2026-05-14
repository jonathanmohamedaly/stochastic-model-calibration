import numpy as np
from scipy.optimize import minimize

from src.calibration.objective import sabr_smile_mse_loss
from src.models.sabr import SABRModel

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
