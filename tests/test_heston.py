import pytest
from math import exp
import numpy as np

from src.calibration.optimizers import calibrate_heston_surface
from src.models.heston import HestonModel
from src.pricing.heston_pricer import (
    heston_call_price,
    heston_put_price,
    heston_characteristic_function,
)
from src.pricing.heston_pricer import heston_implied_volatility


def test_heston_model_valid_parameters():
    model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    assert model.kappa == 2.0
    assert model.theta == 0.04
    assert model.sigma == 0.3
    assert model.rho == -0.7
    assert model.v0 == 0.04


def test_heston_model_invalid_kappa():
    with pytest.raises(ValueError, match="kappa"):
        HestonModel(kappa=0.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)


def test_heston_model_invalid_theta():
    with pytest.raises(ValueError, match="theta"):
        HestonModel(kappa=2.0, theta=0.0, sigma=0.3, rho=-0.7, v0=0.04)


def test_heston_model_invalid_sigma():
    with pytest.raises(ValueError, match="sigma"):
        HestonModel(kappa=2.0, theta=0.04, sigma=0.0, rho=-0.7, v0=0.04)


def test_heston_model_invalid_rho():
    with pytest.raises(ValueError, match="rho"):
        HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-1.0, v0=0.04)


def test_heston_model_invalid_v0():
    with pytest.raises(ValueError, match="v0"):
        HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.0)


def test_heston_feller_condition_true():
    model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    assert model.feller_condition is True


def test_heston_feller_condition_false():
    model = HestonModel(
        kappa=1.0,
        theta=0.04,
        sigma=0.8,
        rho=-0.7,
        v0=0.04,
    )

    assert model.feller_condition is False

def test_heston_characteristic_function_at_zero_is_one():
    model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    cf = heston_characteristic_function(
        u=0.0,
        spot=100,
        maturity=1.0,
        rate=0.03,
        dividend_yield=0.0,
        model=model,
    )

    assert cf == pytest.approx(1.0 + 0.0j, abs=1e-10)


def test_heston_call_price_positive():
    model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    price = heston_call_price(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.03,
        model=model,
    )

    assert price > 0


def test_heston_put_call_parity():
    model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    spot = 100
    strike = 100
    maturity = 1.0
    rate = 0.03

    call = heston_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        model=model,
    )

    put = heston_put_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        model=model,
    )

    lhs = call - put
    rhs = spot - strike * exp(-rate * maturity)

    assert lhs == pytest.approx(rhs, rel=1e-5)

def test_heston_implied_volatility_positive():
    model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    iv = heston_implied_volatility(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.03,
        model=model,
    )

    assert iv > 0

def test_heston_calibration_recovers_synthetic_surface():
    true_model = HestonModel(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        v0=0.04,
    )

    spot = 100.0
    rate = 0.03

    strikes = np.array([90, 100, 110], dtype=float)
    maturities = np.array([0.5, 1.0], dtype=float)

    market_vol_surface = np.zeros((len(maturities), len(strikes)))

    for i, maturity in enumerate(maturities):
        for j, strike in enumerate(strikes):
            market_vol_surface[i, j] = heston_implied_volatility(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                model=true_model,
            )

    calibrated_model = calibrate_heston_surface(
        strikes=strikes,
        maturities=maturities,
        market_vol_surface=market_vol_surface,
        spot=spot,
        rate=rate,
        initial_guess=(1.8, 0.05, 0.25, -0.6, 0.05),
    )

    assert calibrated_model.theta == pytest.approx(true_model.theta, rel=0.3)
    assert calibrated_model.v0 == pytest.approx(true_model.v0, rel=0.3)
    assert calibrated_model.rho == pytest.approx(true_model.rho, rel=0.3)