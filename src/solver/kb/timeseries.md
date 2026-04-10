# Time Series Knowledge Card

## Cross-validation — NEVER shuffle

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```
Each fold trains on data up to time T, validates on (T, T+gap].

## Models

**Tabularised GBDT** (most reliable): build lag/rolling features, then LightGBM/CatBoost.

```python
df['lag_1'] = df.groupby('series_id')['target'].shift(1)
df['lag_7'] = df.groupby('series_id')['target'].shift(7)
df['roll_mean_7'] = df.groupby('series_id')['target'].shift(1).rolling(7).mean()
df['roll_std_7'] = df.groupby('series_id')['target'].shift(1).rolling(7).std()
df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
```

**Statistical**: ARIMA, ExponentialSmoothing (statsmodels).

## Feature engineering

- **Lag features**: shift target by 1, 7, 14, 28 — use `.shift(1)` BEFORE rolling to avoid leakage.
- **Rolling stats**: mean, std, min, max over 7/14/28-day windows.
- **Datetime expansion**: dayofweek, month, quarter, dayofyear, is_weekend.
- **Fourier features**: `sin(2*pi*doy/365.25)`, `cos(...)` for seasonality.
- **Difference features**: target - target.shift(1) for trend removal.

## Pitfalls

- Never shuffle time-ordered data.
- Never compute rolling stats without `.shift(1)` — leaks future.
- Never validate on the wrong horizon — match val horizon to test horizon.
- Direct forecasting (one model per horizon step) usually beats recursive.
