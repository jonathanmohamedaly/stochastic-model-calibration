import numpy as np
import pytest

from src.models.hull_white import HullWhiteModel
from src.pricing.swaption_pricer import (
    discount_factor,
    hull_white_B,
    hull_white_bond_option_volatility,
    zero_coupon_bond_option_price,
    forward_swap_rate,
    swap_annuity,
    hull_white_swaption_volatility,
    payer_swaption_price,
    receiver_swaption_price,
)
from src.calibration.optimizers import calibrate_hull_white_swaption_surface

def test_hull_white_model_valid_parameters():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    assert model.mean_reversion == 0.05
    assert model.volatility == 0.01


def test_hull_white_model_invalid_mean_reversion():
    with pytest.raises(ValueError, match="mean_reversion"):
        HullWhiteModel(mean_reversion=0.0, volatility=0.01)


def test_hull_white_model_invalid_volatility():
    with pytest.raises(ValueError, match="volatility"):
        HullWhiteModel(mean_reversion=0.05, volatility=0.0)

def test_discount_factor():
    df = discount_factor(rate=0.03, maturity=1.0)

    assert df == pytest.approx(0.9704455, rel=1e-6)


def test_hull_white_B_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    B = hull_white_B(t=1.0, T=5.0, model=model)

    assert B > 0


def test_hull_white_bond_option_volatility_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    vol = hull_white_bond_option_volatility(
        option_maturity=1.0,
        bond_maturity=5.0,
        model=model,
    )

    assert vol > 0


def test_zero_coupon_bond_call_price_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    price = zero_coupon_bond_option_price(
        option_type="call",
        strike=0.85,
        option_maturity=1.0,
        bond_maturity=5.0,
        rate=0.03,
        model=model,
    )

    assert price > 0


def test_zero_coupon_bond_put_price_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    price = zero_coupon_bond_option_price(
        option_type="put",
        strike=0.90,
        option_maturity=1.0,
        bond_maturity=5.0,
        rate=0.03,
        model=model,
    )

    assert price >= 0

def test_forward_swap_rate_positive():
    rate = 0.03

    swap_rate = forward_swap_rate(
        rate=rate,
        option_maturity=1.0,
        swap_maturity=5.0,
        payment_frequency=1,
    )

    assert swap_rate > 0


def test_swap_annuity_positive():
    annuity = swap_annuity(
        rate=0.03,
        option_maturity=1.0,
        swap_maturity=5.0,
        payment_frequency=1,
    )

    assert annuity > 0


def test_hull_white_swaption_volatility_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    vol = hull_white_swaption_volatility(
        option_maturity=1.0,
        swap_maturity=5.0,
        model=model,
    )

    assert vol > 0


def test_payer_swaption_price_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    price = payer_swaption_price(
        strike=0.03,
        option_maturity=1.0,
        swap_maturity=5.0,
        rate=0.03,
        model=model,
        payment_frequency=1,
    )

    assert price > 0


def test_receiver_swaption_price_positive():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    price = receiver_swaption_price(
        strike=0.03,
        option_maturity=1.0,
        swap_maturity=5.0,
        rate=0.03,
        model=model,
        payment_frequency=1,
    )

    assert price > 0


def test_swaption_put_call_parity():
    model = HullWhiteModel(mean_reversion=0.05, volatility=0.01)

    strike = 0.03
    option_maturity = 1.0
    swap_maturity = 5.0
    rate = 0.03

    payer = payer_swaption_price(
        strike=strike,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        rate=rate,
        model=model,
    )

    receiver = receiver_swaption_price(
        strike=strike,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
        rate=rate,
        model=model,
    )

    forward_rate = forward_swap_rate(
        rate=rate,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
    )

    annuity = swap_annuity(
        rate=rate,
        option_maturity=option_maturity,
        swap_maturity=swap_maturity,
    )

    parity = annuity * (forward_rate - strike)

    assert payer - receiver == pytest.approx(parity, rel=1e-6)

def test_hull_white_calibration_recovers_synthetic_parameters():
    true_model = HullWhiteModel(
        mean_reversion=0.05,
        volatility=0.01,
    )

    option_maturities = np.array([1.0, 2.0, 3.0])
    swap_maturities = np.array([5.0, 7.0, 10.0])

    rate = 0.03
    strike = 0.03

    market_prices = np.zeros(
        (len(option_maturities), len(swap_maturities))
    )

    for i, option_maturity in enumerate(option_maturities):
        for j, swap_maturity in enumerate(swap_maturities):
            market_prices[i, j] = payer_swaption_price(
                strike=strike,
                option_maturity=option_maturity,
                swap_maturity=swap_maturity,
                rate=rate,
                model=true_model,
            )

    calibrated_model = calibrate_hull_white_swaption_surface(
        option_maturities=option_maturities,
        swap_maturities=swap_maturities,
        market_prices=market_prices,
        strike=strike,
        rate=rate,
        initial_guess=(0.04, 0.012),
    )

    assert calibrated_model.mean_reversion == pytest.approx(
        true_model.mean_reversion,
        rel=0.3,
    )

    assert calibrated_model.volatility == pytest.approx(
        true_model.volatility,
        rel=0.3,
    )