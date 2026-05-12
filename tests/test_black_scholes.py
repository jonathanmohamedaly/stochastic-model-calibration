import pytest
from math import exp

from src.pricing.black_scholes import (
    Black_Scholes_Price,
    Black_Scholes_Vega,
    implied_volatility,
    implied_volatility_brent,
)


def test_call_price_known_value():
    price = Black_Scholes_Price(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="call",
    )

    assert price == pytest.approx(10.4506, rel=1e-4)


def test_put_price_known_value():
    price = Black_Scholes_Price(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="put",
    )

    assert price == pytest.approx(5.5735, rel=1e-4)


def test_put_call_parity():
    spot = 100
    strike = 100
    maturity = 1.0
    rate = 0.05
    volatility = 0.2

    call = Black_Scholes_Price(
        spot, strike, maturity, rate, volatility, option_type="call"
    )
    put = Black_Scholes_Price(
        spot, strike, maturity, rate, volatility, option_type="put"
    )

    lhs = call - put
    rhs = spot - strike * exp(-rate * maturity)

    assert lhs == pytest.approx(rhs, rel=1e-6)


def test_vega_positive():
    vega = Black_Scholes_Vega(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )

    assert vega > 0


def test_implied_volatility_recovers_input_vol():
    true_vol = 0.25

    market_price = Black_Scholes_Price(
        spot=100,
        strike=105,
        maturity=0.75,
        rate=0.03,
        volatility=true_vol,
        option_type="call",
    )

    estimated_vol = implied_volatility(
        market_price=market_price,
        spot=100,
        strike=105,
        maturity=0.75,
        rate=0.03,
        option_type="call",
    )

    assert estimated_vol == pytest.approx(true_vol, rel=1e-6)

def test_implied_volatility_recovers_input_vol_brent():
    true_vol = 0.25

    market_price = Black_Scholes_Price(
        spot=100,
        strike=105,
        maturity=0.75,
        rate=0.03,
        volatility=true_vol,
        option_type="call",
    )

    estimated_vol = implied_volatility_brent(
        market_price=market_price,
        spot=100,
        strike=105,
        maturity=0.75,
        rate=0.03,
        option_type="call",
    )

    assert estimated_vol == pytest.approx(true_vol, rel=1e-6)


def test_market_price_below_intrinsic_raises_error():
    with pytest.raises(ValueError):
        implied_volatility(
            market_price=0.01,
            spot=100,
            strike=50,
            maturity=1.0,
            rate=0.0,
            option_type="call",
        )