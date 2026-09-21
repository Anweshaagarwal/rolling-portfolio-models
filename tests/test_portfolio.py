import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_project import cumulative_return_pct, max_drawdown_pct, ols_weights  # noqa: E402
from regularized_models import lasso_weights, ridge_weights, synthetic_demo  # noqa: E402


class PortfolioMathTest(unittest.TestCase):
    def test_ols_weights_are_fully_invested(self):
        rng = np.random.default_rng(5101)
        returns = rng.normal(0.02, 1.0, size=(126, 8))
        weights = ols_weights(returns)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertEqual(weights.shape, (8,))

    def test_compounding_is_not_simple_addition(self):
        self.assertAlmostEqual(cumulative_return_pct(np.array([10.0, -10.0])), -1.0)

    def test_drawdown_detects_peak_to_trough_loss(self):
        drawdown = max_drawdown_pct(np.array([10.0, -20.0, 5.0]))
        self.assertAlmostEqual(drawdown, -20.0)

    def test_regularized_weights_are_fully_invested(self):
        rng = np.random.default_rng(11)
        returns = rng.normal(size=(126, 8))
        self.assertAlmostEqual(float(ridge_weights(returns, 1.0).sum()), 1.0)
        self.assertAlmostEqual(float(lasso_weights(returns, 0.05).sum()), 1.0)

    def test_synthetic_demo_is_deterministic(self):
        self.assertEqual(synthetic_demo(), synthetic_demo())


if __name__ == "__main__":
    unittest.main()
