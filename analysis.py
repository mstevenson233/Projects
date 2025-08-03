import matplotlib as plt
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

def Analyse_Performance(df):

    total_return = df['cumulative_return'].iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days
    annualised_return = (1 + total_return) ** (365 / days) - 1
    annualised_vol = df['strategy_return'].std() * (252 ** 0.5)
    sharpe_ratio = annualised_return / annualised_vol if annualised_vol != 0 else float('nan')
    running_max = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = (df['strategy_return'] > 0).mean()

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualised_return:.2%}")
    print(f"Annualized Volatility: {annualised_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    return df

# V2

def Compare_With_Benchmark(portfolio_df, benchmark_ticker="SPY", start=None, end=None, initial_capital=1000.0):
    import yfinance as yf
    import matplotlib.pyplot as plt

    # Download SPY data
    benchmark = yf.download(benchmark_ticker, start=start, end=end)['Close']
    benchmark = benchmark.ffill()  # Handle missing data

    # Compute benchmark performance
    benchmark_returns = benchmark.pct_change().fillna(0)
    benchmark_cumulative = (1 + benchmark_returns).cumprod() * initial_capital

    # If multiple assets, sum to get total portfolio value
    if isinstance(portfolio_df, pd.DataFrame):
        portfolio_total = portfolio_df.sum(axis=1)
    else:
        portfolio_total = portfolio_df  # assume already total

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_total, label="Strategy Portfolio")
    plt.plot(benchmark_cumulative, label=f"{benchmark_ticker} (Benchmark)", linestyle='--')
    plt.title("Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return benchmark, benchmark_cumulative

def Calculate_Performance_Metrics(portfolio_value, risk_free_rate=0.0):
    returns = portfolio_value.pct_change().dropna()
    
    cumulative_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    annual_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0])**(252 / len(portfolio_value)) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

def calculate_alpha_beta(strategy_returns, benchmark_returns):
    # Align data
    returns = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strategy = returns.iloc[:,0]
    benchmark = returns.iloc[:,1]

    # Add constant for intercept (alpha)
    X = sm.add_constant(benchmark)
    model = sm.OLS(strategy, X).fit()

    alpha = model.params['const']
    beta = model.params[benchmark.name]

    return alpha, beta