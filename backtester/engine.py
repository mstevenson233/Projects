import matplotlib as plt
import pandas as pd
import numpy as np
import yfinance as yf

def Backtest(df, initial_capital: float = 1000.0):
    df=df.copy()
    df["daily_return"]= df["Close"].pct_change()
    df["strategy_return"]= df["Signal"].shift(1) * df["daily_return"]
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['portfolio_value'] = initial_capital * df['cumulative_return']
    return df


def Backtest_V2(prices_df, signals_df, initial_capital: float = 1000.0):
    returns = prices_df.pct_change()
    strategy_returns = signals_df.shift(1) * returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    cumulative_returns.iloc[0] = 1
    portfolio_value = initial_capital * cumulative_returns
    result = portfolio_value.copy()
    result.columns.name = 'Portfolio Value'
    return result
