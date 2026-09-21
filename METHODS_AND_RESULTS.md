# Revised portfolio baseline

This version applies the professor's updated instruction to delete both
`ME9 BM10` and `BIG HiBM` from the full sample.

## Authoritative specification

- Data block: Average Equal Weighted Returns, Daily
- Assets: 98 after removing `ME9 BM10` and `BIG HiBM`
- Returns: source values are already daily percentage returns
- Missing values: convert `-99.99` and `-999` to `NaN`; do not impute
- Estimation: strict rolling window of 126 prior trading days
- Validation: 2 January 2026 through 30 June 2026, 123 trading days
- Constraint: weights sum to one; short selling is allowed
- OLS: include an intercept
- Risk-free rate: zero

## Revised validation results

| Metric | Equal weight | OLS / minimum variance |
|---|---:|---:|
| Mean daily return (%) | 0.132185 | -0.015073 |
| Daily standard deviation (%) | 0.997378 | 0.877024 |
| Daily Sharpe ratio | 0.132533 | -0.017187 |
| Annualized Sharpe ratio | 2.103890 | -0.272833 |
| Cumulative return (%) | 16.932832 | -2.296576 |
| Maximum drawdown (%) | -8.631955 | -8.988186 |

The transformed OLS and direct minimum-variance implementations agree to
floating-point precision. OLS lowered realized volatility, but its realized
mean return was slightly negative, so it had a negative out-of-sample Sharpe
ratio and underperformed equal weighting in this validation period.

## Rolling results using data from 2023 onward

The first 126 trading days, from 3 January through 5 July 2023, are used to
estimate the first portfolio. The first out-of-sample return is 6 July 2023.
The window then moves forward one trading day at a time through 30 June 2026.

| Metric | Equal weight | Rolling OLS / minimum variance |
|---|---:|---:|
| Out-of-sample days | 749 | 749 |
| Mean daily return (%) | 0.070707 | 0.064289 |
| Daily standard deviation (%) | 1.192047 | 1.251436 |
| Daily Sharpe ratio | 0.059316 | 0.051372 |
| Annualized Sharpe ratio | 0.941607 | 0.815505 |
| Cumulative return (%) | 61.032438 | 52.653768 |
| Maximum drawdown (%) | -24.157830 | -17.773306 |

The cumulative return is compounded rather than added. Starting with wealth
of 1, each day updates wealth using

`wealth_t = wealth_(t-1) * (1 + return_t / 100)`.

The cumulative return reported on each date is

`100 * (wealth_t - 1)`.

For example, a 10% gain followed by a 10% loss leaves wealth at
`1 * 1.10 * 0.90 = 0.99`, so the cumulative return is -1%, not 0%.

## Training-only LASSO and Ridge results

LASSO and Ridge were developed only with data from 3 January 2023 through 31
December 2025. No 2026 validation return was used. Each rolling fit uses the
previous 126 trading days and selects alpha from 101 values between `1e-8` and
`1e8` using five time-ordered cross-validation folds inside that window.

| Metric | Equal weight | Rolling OLS | Rolling LASSO | Rolling Ridge |
|---|---:|---:|---:|---:|
| Training-era test days | 626 | 626 | 626 | 626 |
| Mean daily return (%) | 0.058628 | 0.079882 | 0.045396 | 0.050342 |
| Daily standard deviation (%) | 1.227011 | 1.312507 | 0.759192 | 0.728296 |
| Daily Sharpe ratio | 0.047781 | 0.060862 | 0.059795 | 0.069123 |
| Annualized Sharpe ratio | 0.758496 | 0.966161 | 0.949219 | 1.097287 |
| Cumulative return (%) | 37.713621 | 56.241984 | 30.481127 | 34.773340 |
| Maximum drawdown (%) | -24.157830 | -17.773306 | -15.148219 | -12.126119 |

LASSO reduced volatility, drawdown, turnover, and extreme weights relative to
OLS. Its training-era Sharpe ratio was higher than equal weighting but slightly
lower than OLS. These are model-development results, not 2026 validation
results.

Ridge produced the highest training-era Sharpe ratio and the lowest volatility
and drawdown. Its cumulative return was below OLS but above LASSO. These are
still training-period model-development results and do not establish how Ridge
will perform on the untouched 2026 validation period.

The final training fit uses 3 July through 31 December 2025. It selected alpha
`0.0173780083`, retained 38 nonzero beta coefficients, produced weights that
sum to 1, and did not calculate any 2026 return.

The final Ridge fit uses the same dates. It selected alpha `19.0546071796`,
produced weights that sum to 1, and did not calculate any 2026 return.

## Equations

Let `R` be the 126 by 98 matrix of daily percentage returns and let `w` be the
98 by 1 portfolio-weight vector.

The fully invested minimum-variance problem is

`min_w w' Sigma w`, subject to `1'w = 1`.

Equal weights are `w_EW = (1/p)1`, where `p = 98`. Define
`N = [I_(p-1); -1']`. Every fully invested portfolio can be written as

`w = w_EW - N beta`.

Set `y = R w_EW` and `X = R N`. The OLS model is

`y_t = alpha + x_t' beta + epsilon_t`,

and estimates `alpha` and `beta` by minimizing

`sum_t epsilon_t^2 = ||y - alpha*1 - X beta||_2^2`.

After estimating `beta`, the portfolio weights are recovered using
`w_hat = w_EW - N beta_hat`. Because `1'N = 0`, these weights sum to one.
With an intercept, the residual is

`epsilon_t = r_t' w_hat - alpha_hat`.

The fitted intercept is the sample mean portfolio return, so the residuals are
the portfolio returns around their mean. Their squared sum is proportional to
sample portfolio variance. This is why the OLS solution and the direct
minimum-variance solution coincide.

The error term here is not a stock-price forecasting error. It is the daily
deviation of the constructed portfolio return from its in-sample mean. A
separate issue is estimation error: 98 weights are estimated from only 126
days, which can produce unstable, highly leveraged weights.

## What each model optimizes

- Equal weight does not estimate or optimize anything. Every asset receives
  `1/98`.
- Naive OLS minimizes in-sample squared residuals, which is equivalent here to
  minimizing in-sample portfolio variance under the full-investment constraint.
  It does not maximize expected return or Sharpe ratio.
- Direct minimum variance minimizes `w' Sigma_hat w` subject to `1'w = 1`.
  It is a numerical cross-check of the OLS transformation.
- LASSO minimizes OLS loss plus `lambda ||beta||_1`. It is implemented on
  the training period only.
- Ridge minimizes OLS loss plus `lambda ||beta||_2^2`. It is implemented on
  the training period only.

Both penalties target estimation error and weight instability.
