"""Forecasting models and helpers."""

import numpy as np


def forecast_placeholder(series, periods=12):
    """Return a naive forecasting placeholder (last-value)."""
    last = series.iloc[-1]
    return np.repeat(last, periods)
