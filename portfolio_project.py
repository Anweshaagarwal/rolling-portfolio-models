#!/usr/bin/env python3
"""Portfolio optimization project: data cleaning and rolling naive OLS.

The source is the Fama-French 100 portfolios daily CSV. This script keeps only
the Average Equal Weighted Returns block, removes ME9 BM10 and BIG HiBM as
required, maps the source missing-value sentinels to NaN, and runs the OLS
transformation described in the project specification.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np


MISSING = {-99.99, -999.0}
DROP_ASSETS = ("ME9 BM10", "BIG HiBM")
WINDOW = 126
VALIDATION_START, VALIDATION_END = "20260102", "20260630"
TEST_START, TEST_END = "20260801", "20260830"


def load_equal_weighted_block(source: Path):
    """Return dates, asset labels, and equal-weighted returns from block 2."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one CSV in {source}, found {len(names)}")
            with zf.open(names[0]) as f:
                rows = list(csv.reader(line.decode("utf-8") for line in f))
    else:
        with source.open(newline="") as f:
            rows = list(csv.reader(f))

    # Block boundaries are stable in the supplied file: the second header is
    # the header immediately after "Average Equal Weighted Returns -- Daily".
    label_row = next(i for i, row in enumerate(rows)
                     if row and "Average Equal Weighted Returns" in row[0])
    header = rows[label_row + 1]
    data = []
    for row in rows[label_row + 2:]:
        # A blank separator terminates this block.  Do not accidentally join
        # the following firm-count and firm-size datasets.
        if len(row) != len(header):
            break
        if row[0].strip().isdigit():
            data.append(row)
    labels = [x.strip() for x in header[1:]]
    dates = [row[0].strip() for row in data]
    values = np.array([[float(x.strip()) for x in row[1:]] for row in data])

    values[np.isin(values, list(MISSING))] = np.nan
    missing_drop_labels = [label for label in DROP_ASSETS if label not in labels]
    if missing_drop_labels:
        raise ValueError(f"Required assets not found: {missing_drop_labels}")
    drop_indices = {labels.index(label) for label in DROP_ASSETS}
    keep = np.array([i not in drop_indices for i in range(values.shape[1])])
    return dates, [labels[i] for i in range(len(labels)) if keep[i]], values[:, keep]


def save_cleaned_csv(path: Path, dates, labels, values):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", *labels])
        for date, row in zip(dates, values):
            writer.writerow([date, *("" if np.isnan(x) else f"{x:.6f}" for x in row)])


def ols_weights(window_returns: np.ndarray) -> np.ndarray:
    """Fit y = alpha + X beta and convert beta to a fully invested portfolio."""
    p = window_returns.shape[1]
    w_ew = np.full(p, 1.0 / p)
    # N = [I_(p-1); -1^T], so w = w_EW - N beta and X = R N.
    n = np.vstack([np.eye(p - 1), -np.ones((1, p - 1))])
    y = window_returns @ w_ew
    x = window_returns @ n
    design = np.column_stack([np.ones(len(y)), x])
    alpha_beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta = alpha_beta[1:]
    w = w_ew - n @ beta
    return w / w.sum()


def daily_sharpe(returns: np.ndarray) -> float:
    """Daily Sharpe ratio with the assignment's zero risk-free rate."""
    return float(np.mean(returns) / np.std(returns, ddof=1))


def annualized_sharpe(returns: np.ndarray) -> float:
    return float(np.sqrt(252) * daily_sharpe(returns))


def cumulative_return_pct(returns: np.ndarray) -> float:
    return float((np.prod(1.0 + returns / 100.0) - 1.0) * 100.0)


def max_drawdown_pct(returns: np.ndarray) -> float:
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns / 100.0)))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return float(100.0 * np.min(drawdown))


def run_backtest(dates, values):
    date_to_i = {d: i for i, d in enumerate(dates)}
    available = []
    for date in dates:
        if (VALIDATION_START <= date <= VALIDATION_END) or (TEST_START <= date <= TEST_END):
            available.append(date)

    rows = []
    p = values.shape[1]
    w_ew = np.full(p, 1.0 / p)
    for date in available:
        i = date_to_i[date]
        if i < WINDOW:
            continue
        train = values[i - WINDOW:i]
        target = values[i]
        if not (np.isfinite(train).all() and np.isfinite(target).all()):
            rows.append({"date": date, "status": "skipped_missing_data"})
            continue
        w = ols_weights(train)
        rows.append({
            "date": date,
            "status": "ok",
            "ew_return_pct": float(target @ w_ew),
            "ols_return_pct": float(target @ w),
            "weight_sum": float(w.sum()),
            "max_abs_weight": float(np.max(np.abs(w))),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dates, labels, values = load_equal_weighted_block(args.source)
    cleaned = args.outdir / "cleaned_equal_weighted_daily.csv"
    save_cleaned_csv(cleaned, dates, labels, values)

    results = run_backtest(dates, values)
    result_csv = args.outdir / "naive_ols_backtest.csv"
    with result_csv.open("w", newline="") as f:
        fields = ["date", "status", "ew_return_pct", "ols_return_pct", "weight_sum", "max_abs_weight"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if r["status"] == "ok"]
    val = [r for r in ok if VALIDATION_START <= r["date"] <= VALIDATION_END]
    test = [r for r in ok if TEST_START <= r["date"] <= TEST_END]
    ew_val = np.array([r["ew_return_pct"] for r in val])
    ols_val = np.array([r["ols_return_pct"] for r in val])
    summary = {
        "source": str(args.source),
        "block_kept": "Average Equal Weighted Returns -- Daily",
        "n_observations": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "n_assets_after_cleaning": len(labels),
        "dropped_assets": list(DROP_ASSETS),
        "missing_sentinels": sorted(MISSING),
        "missing_values_after_drop": int(np.isnan(values).sum()),
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "validation_n_days": len(val),
        "rolling_window_days": WINDOW,
        "ols_intercept": True,
        "short_selling": True,
        "risk_free_rate": 0,
        "validation_ew_mean_pct": float(np.mean(ew_val)) if val else None,
        "validation_ew_std_pct": float(np.std(ew_val, ddof=1)) if val else None,
        "validation_ew_daily_sharpe": daily_sharpe(ew_val) if val else None,
        "validation_ew_annualized_sharpe": annualized_sharpe(ew_val) if val else None,
        "validation_ew_cumulative_return_pct": cumulative_return_pct(ew_val) if val else None,
        "validation_ew_max_drawdown_pct": max_drawdown_pct(ew_val) if val else None,
        "validation_ols_mean_pct": float(np.mean(ols_val)) if val else None,
        "validation_ols_std_pct": float(np.std(ols_val, ddof=1)) if val else None,
        "validation_ols_daily_sharpe": daily_sharpe(ols_val) if val else None,
        "validation_ols_annualized_sharpe": annualized_sharpe(ols_val) if val else None,
        "validation_ols_cumulative_return_pct": cumulative_return_pct(ols_val) if val else None,
        "validation_ols_max_drawdown_pct": max_drawdown_pct(ols_val) if val else None,
        "test_period": [TEST_START, TEST_END],
        "test_n_days_available": len(test),
        "test_status": "unavailable in supplied file" if not test else "available",
        "first_ols_weight_sum": ok[0]["weight_sum"] if ok else None,
    }
    (args.outdir / "data_cleaning_and_ols_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
