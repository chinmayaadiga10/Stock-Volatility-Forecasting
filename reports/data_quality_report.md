# Data Quality & Validation Report

**Project:** Stock Market Volatility Forecasting Using Deep Learning: A Comparative Analysis of LSTM, GRU and Transformer Networks  
**Scope:** Stage 2 — Data Collection and Validation  
**Target Stocks:** RELIANCE.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, TCS.NS  
**Historical Horizon:** 2015-01-01 to 2026-11-17  

---

## 1. Summary of Collected Datasets

| Ticker | Trading Days | Start Date | End Date | Missing OHLCV | Duplicate Dates | Invalid Prices (<=0) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RELIANCE.NS** | 2932 | 2015-01-01 | 2026-11-17 | 0 | 0 | 0 |
| **HDFCBANK.NS** | 2932 | 2015-01-01 | 2026-11-17 | 0 | 0 | 0 |
| **INFY.NS** | 2932 | 2015-01-01 | 2026-11-17 | 0 | 0 | 0 |
| **ICICIBANK.NS** | 2932 | 2015-01-01 | 2026-11-17 | 0 | 0 | 0 |
| **TCS.NS** | 2932 | 2015-01-01 | 2026-11-17 | 0 | 0 | 0 |

---

## 2. Trading Calendar Consistency Across Stocks

- **Total Unique Trading Dates Across All 5 Stocks:** 2932
- **Synchronized Common Trading Dates:** 2932
- **Alignment Ratio:** 100.00%

All 5 stocks are constituent members of the NIFTY 50 index traded on the National Stock Exchange of India (NSE). The dates are virtually 100% aligned, providing a unified historical time series for comparative benchmarking.

---

## 3. Statistical Properties of Daily Log Returns

| Ticker | Mean Return (%) | Daily Std Dev (%) | Skewness | Excess Kurtosis | Jarque-Bera p-value | Stylized Fact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RELIANCE.NS** | 0.067% | 1.697% | 0.165 | 8.52 | 0.0e+00 | Leptokurtic / Fat-Tailed |
| **HDFCBANK.NS** | 0.046% | 1.409% | -0.299 | 9.84 | 0.0e+00 | Leptokurtic / Fat-Tailed |
| **INFY.NS** | 0.040% | 1.702% | -0.499 | 8.82 | 0.0e+00 | Leptokurtic / Fat-Tailed |
| **ICICIBANK.NS** | 0.058% | 1.883% | -0.091 | 9.53 | 0.0e+00 | Leptokurtic / Fat-Tailed |
| **TCS.NS** | 0.031% | 1.509% | -0.076 | 3.78 | 0.0e+00 | Leptokurtic / Fat-Tailed |

> [!NOTE]
> **Stylized Facts Observed:**
> 1. **Fat Tails (Leptokurtosis):** All stocks exhibit excess kurtosis significantly greater than 0 (normal distribution kurtosis = 3, excess = 0). Jarque-Bera tests reject normality at p < 0.001.
> 2. **Volatility Clustering:** Large price movements are followed by large price movements, and small movements by small movements. This strongly justifies conditional volatility modeling (GARCH and Deep Learning sequence models).

---

## 4. Forecasting Target Properties: 5-Day Forward Realized Volatility

$$\sigma_{t, t+5} = \sqrt{\sum_{k=1}^5 r_{t+k}^2}$$

| Ticker | Mean 5d Vol | Median 5d Vol | 95th Percentile | Max 5d Vol | Ann. Mean Vol (%) | Ann. Median Vol (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RELIANCE.NS** | 0.0327 | 0.0288 | 0.0621 | 0.2379 | 23.21% | 20.41% |
| **HDFCBANK.NS** | 0.0263 | 0.0221 | 0.0546 | 0.1816 | 18.70% | 15.72% |
| **INFY.NS** | 0.0330 | 0.0289 | 0.0650 | 0.1824 | 23.41% | 20.50% |
| **ICICIBANK.NS** | 0.0353 | 0.0297 | 0.0743 | 0.2260 | 25.05% | 21.06% |
| **TCS.NS** | 0.0299 | 0.0266 | 0.0581 | 0.1385 | 21.21% | 18.88% |

> [!IMPORTANT]
> **Causality & Data Leakage Prevention:**
> Input features at trading day $t$ utilize exclusively retrospective information (prices, volumes, and returns up to date $t$). The forecasting target $\sigma_{t, t+5}$ strictly aggregates forward returns $r_{t+1}$ through $r_{t+5}$. The terminal 5 records of each dataset naturally hold `NaN` for this target and are reserved for future out-of-sample forward deployment.

---

## 5. Artifacts and Generated Visualizations

- **[price_history.png](file://reports/figures/price_history.png):** Multi-year closing price trajectories (2015–2026).
- **[daily_log_returns.png](file://reports/figures/daily_log_returns.png):** Daily return time series highlighting volatility spikes (e.g. March 2020 COVID shock).
- **[forward_volatility_5d.png](file://reports/figures/forward_volatility_5d.png):** Target annualized 5-day forward realized volatility series.
- **[return_distributions_kde.png](file://reports/figures/return_distributions_kde.png):** Empirical vs Normal probability densities.
- **[volatility_distributions_kde.png](file://reports/figures/volatility_distributions_kde.png):** Realized volatility target distributions.
- **[cross_stock_correlation.png](file://reports/figures/cross_stock_correlation.png):** Inter-stock correlations across returns and volatility.
