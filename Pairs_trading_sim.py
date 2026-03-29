# %% Relevant libraries
import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %%Simulate cointegrated stock price series as random walks w. noise
n = 1000
walk = np.zeros(n, dtype=int)          
walk[0] = 100

steps = np.random.choice([-1, 1], size=n-1)

for t in range(1, n):                  
    walk[t] = walk[t-1] + steps[t-1]

X = walk + np.random.normal(0, 1, size=n) # Stock X
Y = 0.85*X + np.random.normal(0, 1, size=n) # Stock Y, cointegrated with X

# Plot of simulated stock prices
plt.figure(figsize=(10,5))
plt.plot(X, label='Stock X')
plt.plot(Y, label='Stock Y')
plt.title('Simulated Cointegrated Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#%% Test for cointegration using Engle-Granger method (True coeffecient = 0.85)
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const).fit()
print(model.summary())
residuals = model.resid
adf_results = sm.tsa.adfuller(residuals)
print("ADF Statistic: " +str(round(adf_results[0],2)))
print("p-value: " +str(round(adf_results[1],4)))

# Plot residuals
plt.figure(figsize=(10,5))
plt.plot(residuals, label="Residuals")
plt.title("Residuals from Engle-Granger Regression")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.legend()
plt.show()

#%% Functions for backtesting the pairs trading strategy
def backtest_pairs_strategy(
    X,
    Y,
    lookback=100,
    entry_z=2.0,
    exit_z=0.5,
    tc_bps=5
):
    data = pd.DataFrame({"X": pd.Series(X, dtype=float),"Y": pd.Series(Y, dtype=float)})

    # Calculate log returns
    data["ret_X"] = np.log(data["X"]).diff()
    data["ret_Y"] = np.log(data["Y"]).diff()

    # Initialise columns for stategy
    data["alpha"] = np.nan
    data["beta"] = np.nan
    data["spread"] = np.nan
    data["zscore"] = np.nan
    data["signal"] = np.nan

    current_signal = 0  # 1 = long spread, -1 = short spread, 0 = no trade

    for t in range(lookback, len(data)):
        window = data.iloc[t - lookback:t].copy()

        # Rolling hedge ratio estimated only on past data (Engle-Granger regressions)
        X_reg = sm.add_constant(window["X"])
        model = sm.OLS(window["Y"], X_reg).fit()

        alpha = model.params["const"]
        beta = model.params["X"]

        spread_hist = window["Y"] - alpha - beta * window["X"]
        mu = spread_hist.mean()
        sigma = spread_hist.std()

        spread_t = data.loc[t, "Y"] - alpha - beta * data.loc[t, "X"]
        z_t = (spread_t - mu) / sigma if sigma > 0 else np.nan

        # Trading signals based on rolling z-scores of the spread
        if current_signal == 0:
            if z_t < -entry_z:
                current_signal = 1      # long spread
            elif z_t > entry_z:
                current_signal = -1     # short spread
        elif current_signal == 1:
            if z_t > -exit_z:
                current_signal = 0
        elif current_signal == -1:
            if z_t < exit_z:
                current_signal = 0

        data.loc[t, "alpha"] = alpha
        data.loc[t, "beta"] = beta
        data.loc[t, "spread"] = spread_t
        data.loc[t, "zscore"] = z_t
        data.loc[t, "signal"] = current_signal

    # Trade on next period's return, not same-period return
    data["position"] = data["signal"].shift(1).fillna(0)
    data["beta_trade"] = data["beta"].shift(1)

    # Dollar-normalized weights
    norm = 1 + data["beta_trade"].abs()
    data["w_Y"] = data["position"] * (1 / norm)
    data["w_X"] = data["position"] * (-data["beta_trade"] / norm)

    # Gross strategy return
    data["gross_ret"] = data["w_Y"] * data["ret_Y"] + data["w_X"] * data["ret_X"]

    # Simple turnover-based cost model
    data["turnover"] = data["w_Y"].diff().abs().fillna(0) + data["w_X"].diff().abs().fillna(0)
    data["cost"] = data["turnover"] * (tc_bps / 10000.0)

    data["net_ret"] = data["gross_ret"] - data["cost"]
    data["cum_ret"] = (1 + data["net_ret"].fillna(0)).cumprod()

    return data

def performance_summary(data):
    rets = data["net_ret"].dropna()
    if len(rets) == 0:
        return {}

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    equity = data["cum_ret"].dropna()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    trades = (data["position"].diff().abs() > 0).sum()

    return {
        "Annual return": ann_ret,
        "Annual vol": ann_vol,
        "Sharpe": sharpe,
        "Max drawdown": max_dd,
        "Trades": int(trades)
    }

#%% Running on simulated data
backtest = backtest_pairs_strategy(X, Y, lookback=100, entry_z=2.0, exit_z=0.5, tc_bps=5)
print(performance_summary(backtest))

#%% Plotting the results
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axes[0].plot(backtest["spread"])
axes[0].set_title("Rolling Spread")
axes[1].plot(backtest["zscore"])
axes[1].axhline(2, linestyle="--")
axes[1].axhline(-2, linestyle="--")
axes[1].axhline(0.5, linestyle=":")
axes[1].axhline(-0.5, linestyle=":")
axes[1].set_title("Z-score")
axes[2].plot(backtest["cum_ret"])
axes[2].set_title("Cumulative Net Return")
plt.tight_layout()
plt.show()