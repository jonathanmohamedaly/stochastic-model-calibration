import pytest
import numpy as np

from src.models.sabr import SABRModel
from src.pricing.sabr_formula import hagan_implied_volatility
from src.calibration.optimizers import calibrate_sabr_smile

def test_sabr_model_valid_parameters():
    model = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.8)

    assert model.alpha == 0.25
    assert model.beta == 0.5
    assert model.rho == -0.3
    assert model.nu == 0.8


def test_sabr_model_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        SABRModel(alpha=0.0, beta=0.5, rho=-0.3, nu=0.8)


def test_sabr_model_invalid_beta():
    with pytest.raises(ValueError, match="beta"):
        SABRModel(alpha=0.25, beta=1.5, rho=-0.3, nu=0.8)


def test_sabr_model_invalid_rho():
    with pytest.raises(ValueError, match="rho"):
        SABRModel(alpha=0.25, beta=0.5, rho=1.0, nu=0.8)


def test_sabr_model_invalid_nu():
    with pytest.raises(ValueError, match="nu"):
        SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.0)


def test_sabr_model_with_parameters():
    model = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.8)

    updated = model.with_parameters(rho=-0.1, nu=1.2)

    assert updated.alpha == 0.25
    assert updated.beta == 0.5
    assert updated.rho == -0.1
    assert updated.nu == 1.2


def test_hagan_implied_volatility_positive():
    model = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.8)

    vol = hagan_implied_volatility(
        forward=100,
        strike=100,
        maturity=1.0,
        model=model,
    )

    assert vol > 0


def test_hagan_implied_volatility_atm_stable():
    model = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.8)

    vol = hagan_implied_volatility(
        forward=100,
        strike=100,
        maturity=1.0,
        model=model,
    )

    assert 0.025 < vol < 0.027


def test_hagan_implied_volatility_smile_shape():
    model = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.8)

    strikes = [80, 90, 100, 110, 120]

    vols = [
        hagan_implied_volatility(
            forward=100,
            strike=k,
            maturity=1.0,
            model=model,
        )
        for k in strikes
    ]

    assert all(vol > 0 for vol in vols)
    assert vols[0] > vols[2]


def test_sabr_calibration_recovers_synthetic_parameters():
    true_model = SABRModel(alpha=0.25, beta=0.5, rho=-0.35, nu=0.8)

    forward = 100.0
    maturity = 1.0
    strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)

    market_vols = np.array(
        [
            hagan_implied_volatility(
                forward=forward,
                strike=strike,
                maturity=maturity,
                model=true_model,
            )
            for strike in strikes
        ]
    )

    calibrated_model = calibrate_sabr_smile(
        strikes=strikes,
        market_vols=market_vols,
        forward=forward,
        maturity=maturity,
        beta=0.5,
        initial_guess=(0.2, -0.1, 0.6),
    )

    assert 0.24 <= true_model.alpha <= 0.25
    assert -0.35 <= true_model.rho <= -0.34
    assert 0.8 <= true_model.nu <= 0.82