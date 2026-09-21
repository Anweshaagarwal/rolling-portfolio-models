# Rolling portfolio model validation

This project compares equal weighting with rolling OLS, LASSO, and Ridge portfolios over 98 assets. Every target day uses only the previous 126 trading days. The final comparison uses 123 untouched trading days from 2 January through 30 June 2026.

## Main result

LASSO produced the lowest 2026 maximum drawdown at `-4.60%` and lower volatility than equal weighting. Equal weighting still had the best annualised Sharpe ratio at `2.10` and the highest cumulative return at `16.93%`. The result is a useful warning: an optimisation method can control one risk measure and still lose on the full out-of-sample objective.

| Model | Annualised Sharpe | Cumulative return | Maximum drawdown |
|---|---:|---:|---:|
| Equal weight | 2.10 | 16.93% | -8.63% |
| OLS | -0.27 | -2.30% | -8.99% |
| LASSO | 1.23 | 5.16% | -4.60% |
| Ridge | -0.34 | -1.74% | -7.92% |

## Validation controls

- The target day and all future returns are excluded from each training window.
- LASSO and Ridge select their penalty through time-ordered cross-validation inside each rolling window.
- Every training window contains 126 observations.
- Every weight vector sums to one within numerical tolerance.
- Missing sentinel values are removed rather than imputed.

See `METHODS_AND_RESULTS.md` for equations, assumptions, and the full comparison.

## Reproduce

Run `portfolio_project.py` from a directory containing the course dataset described in `METHODS_AND_RESULTS.md`. The raw dataset is not included because its redistribution terms are unclear. The committed JSON summary and chart let reviewers inspect the reported validation output.

```bash
python3 portfolio_project.py path/to/100_Portfolios_10x10_Daily.csv
python3 regularized_models.py
python3 -m unittest discover -s tests -v
```

`regularized_models.py` contains reviewable NumPy implementations of the Ridge and LASSO transformations plus a deterministic synthetic-data smoke test. The synthetic demo verifies execution without presenting its output as evidence for the reported course-data results.
