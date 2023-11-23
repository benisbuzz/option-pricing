from typing import Optional, Literal
from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
import os
import requests
from scipy.stats import normaltest

FRED_API_KEY = os.getenv("FRED_API_KEY")


def get_fred(series_id: str):
    url = (
        "https://api.stlouisfed.org/fred/series/observations?series_id="
        + series_id
        + "&api_key="
        + FRED_API_KEY
        + "&file_type=json"
    )
    request = requests.get(url)
    if request.status_code != 200:
        raise ValueError(f"invalid series_code: {series_id}\nurl: {url}")
    data = request.json()["observations"]
    idx, values = (
        [pd.Timestamp(ele["date"]) for ele in data],
        [
            float(ele["value"])
            if any(char.isdigit() for char in ele["value"])
            else np.nan
            for ele in data
        ],
    )
    return pd.Series(values, idx)


class Option:
    def __init__(
        self,
        *,
        put_call: Literal["C", "P"],
        spot: float,
        strike: float,
        expiration_date: str,
        price: Optional[float] = None,
        volatility: Optional[float] = None,
    ):
        self.put_call = put_call
        self.spot = spot
        self.strike = strike
        self.expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        self.days_to_expiration = (self.expiration_date - datetime.utcnow()).days
        self.risk_free_rate = get_fred("DGS1MO").iloc[-1] / 100

        if price:
            self.price = price
            self.volatility = self.get_implied_vol(
                self.put_call,
                self.price,
                self.spot,
                self.strike,
                self.expiration_date / 365,
                self.risk_free_rate,
            )

        else:
            self.volatility = volatility
            if put_call == "C":
                self.price = self.get_call_price(
                    self.spot,
                    self.strike,
                    self.days_to_expiration / 365,
                    self.risk_free_rate,
                    self.volatility,
                )
            else:
                self.price = self.get_put_price(
                    self.spot,
                    self.strike,
                    self.days_to_expiration / 365,
                    self.risk_free_rate,
                    self.volatility,
                )

        self.delta = self.get_delta(
            self.put_call,
            self.spot,
            self.strike,
            self.days_to_expiration / 365,
            self.risk_free_rate,
            self.volatility,
        )
        self.theta = self.get_theta(
            self.put_call,
            self.spot,
            self.strike,
            self.days_to_expiration / 365,
            self.risk_free_rate,
            self.volatility,
        )
        self.vega = self.get_vega(
            self.spot,
            self.strike,
            self.days_to_expiration / 365,
            self.risk_free_rate,
            self.volatility,
        )
        self.gamma = self.get_gamma(
            self.spot,
            self.strike,
            self.days_to_expiration / 365,
            self.risk_free_rate,
            self.volatility,
        )
        self.rho = self.get_rho(
            self.put_call,
            self.spot,
            self.strike,
            self.days_to_expiration / 365,
            self.risk_free_rate,
            self.volatility
        )
        self.option_df = pd.Series(
            [self.price, self.volatility, self.delta, self.gamma, self.theta, self.vega, self.rho],
            ["Price", "Implied Vol", "Delta", "Gamma", "Theta", "Vega", "Rho"]
        )

    def get_d1(self, S, K, T, r, sigma):
        return (log(S / K) + (r + sigma**2 / 2) * T) / sigma * sqrt(T)

    def get_d2(self, S, K, T, r, sigma):
        return self.get_d1(S, K, T, r, sigma) - sigma * sqrt(T)

    def get_call_price(self, S, K, T, r, sigma):
        return S * norm.cdf(self.get_d1(S, K, T, r, sigma)) - K * exp(
            -r * T
        ) * norm.cdf(self.get_d2(S, K, T, r, sigma))

    def get_put_price(self, S, K, T, r, sigma):
        return K * exp(-r * T) - S + self.get_call_price(S, K, T, r, sigma)

    def get_delta(self, put_call, S, K, T, r, sigma):
        if put_call == "C":
            return norm.cdf(self.get_d1(S, K, T, r, sigma))
        return -norm.cdf(-self.get_d1(S, K, T, r, sigma))

    def get_gamma(self, S, K, T, r, sigma):
        return norm.pdf(self.get_d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))

    def get_vega(self, S, K, T, r, sigma):
        return 0.01 * (S * norm.pdf(self.get_d1(S, K, T, r, sigma)) * sqrt(T))

    def get_theta(self, put_call, S, K, T, r, sigma):
        if put_call == "C":
            return 0.01 * (
                -(S * norm.pdf(self.get_d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T))
                - r * K * exp(-r * T) * norm.cdf(self.get_d2(S, K, T, r, sigma))
            )
        return 0.01 * (
            -(S * norm.pdf(self.get_d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T))
            + r * K * exp(-r * T) * norm.cdf(-self.get_d2(S, K, T, r, sigma))
        )

    def get_rho(self, put_call, S, K, T, r, sigma):
        if put_call == "C":
            return 0.01 * (
                K * T * exp(-r * T) * norm.cdf(self.get_d2(S, K, T, r, sigma))
            )
        return 0.01 * (-K * T * exp(-r * T) * norm.cdf(-self.get_d2(S, K, T, r, sigma)))

    def get_implied_vol(self, put_call, P, S, K, T, r):
        sigma = 0.001
        if put_call == "C":
            while sigma < 1:
                Price_implied = S * norm.cdf(self.get_d1(S, K, T, r, sigma)) - K * exp(
                    -r * T
                ) * norm.cdf(self.get_d2(S, K, T, r, sigma))
                if P - (Price_implied) < 0.001:
                    return sigma
                sigma += 0.001
            raise RuntimeError("Could not find correct IV")
        while sigma < 1:
            Price_implied = K * exp(-r * T) - S + self.get_call_price(S, K, T, r, sigma)
            if P - (Price_implied) < 0.001:
                return sigma
            sigma += 0.001
        raise RuntimeError("Could not find correct IV")
    
def test_for_normal(series: pd.Series):
    return normaltest(list(series.pct_change()))