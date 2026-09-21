#!/usr/bin/env python3
"""Reviewable Ridge and LASSO portfolio estimators using NumPy only."""

from __future__ import annotations

import numpy as np


def portfolio_regression(window_returns: np.ndarray):
    """Return centered regression data and the full-investment transform."""
    rows, assets = window_returns.shape
    if rows < 3 or assets < 2:
        raise ValueError("Expected at least three rows and two assets")
    equal_weight = np.full(assets, 1.0 / assets)
    transform = np.vstack([np.eye(assets - 1), -np.ones((1, assets - 1))])
    target = window_returns @ equal_weight
    features = window_returns @ transform
    return features - features.mean(axis=0), target - target.mean(), transform, equal_weight


def _weights(beta: np.ndarray, transform: np.ndarray, equal_weight: np.ndarray):
    weights = equal_weight - transform @ beta
    return weights / weights.sum()


def ridge_weights(window_returns: np.ndarray, alpha: float) -> np.ndarray:
    """Estimate fully invested Ridge weights for one rolling window."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    x, y, transform, equal_weight = portfolio_regression(window_returns)
    beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    return _weights(beta, transform, equal_weight)


def _soft_threshold(value: float, penalty: float) -> float:
    return np.sign(value) * max(abs(value) - penalty, 0.0)


def lasso_weights(
    window_returns: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 5_000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Estimate fully invested LASSO weights by coordinate descent."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    x, y, transform, equal_weight = portfolio_regression(window_returns)
    scale = np.linalg.norm(x, axis=0)
    safe_scale = np.where(scale == 0, 1.0, scale)
    standardized = x / safe_scale
    beta_scaled = np.zeros(standardized.shape[1])
    prediction = standardized @ beta_scaled

    for _ in range(max_iter):
        previous = beta_scaled.copy()
        for column in range(standardized.shape[1]):
            feature = standardized[:, column]
            residual = y - prediction + feature * beta_scaled[column]
            updated = _soft_threshold(feature @ residual, alpha)
            prediction += feature * (updated - beta_scaled[column])
            beta_scaled[column] = updated
        if np.max(np.abs(beta_scaled - previous)) < tolerance:
            break

    return _weights(beta_scaled / safe_scale, transform, equal_weight)


def synthetic_demo(seed: int = 7) -> dict[str, float]:
    """Run a deterministic smoke test without the restricted course dataset."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.04, 0.8, size=(160, 1))
    returns = market + rng.normal(0.0, 0.6, size=(160, 8))
    train, test = returns[:126], returns[126:]
    models = {
        "equal_weight": np.full(returns.shape[1], 1.0 / returns.shape[1]),
        "ridge": ridge_weights(train, alpha=10.0),
        "lasso": lasso_weights(train, alpha=0.05),
    }
    return {
        name: float(np.std(test @ weights, ddof=1))
        for name, weights in models.items()
    }


if __name__ == "__main__":
    for name, volatility in synthetic_demo().items():
        print(f"{name}: test volatility = {volatility:.4f}")
