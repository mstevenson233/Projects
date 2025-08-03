from data_loader import Load_Data
from stratedies import Generate_Random_Signals
from engine import Backtest
from analysis import Analyse_Performance
from data_loader import Load_Data_V2
from stratedies import Generate_Random_Signals_V2
from engine import Backtest_V2
from analysis import Analyse_Performance
from analysis import Compare_With_Benchmark
from analysis import Calculate_Performance_Metrics
from analysis import calculate_alpha_beta
import time
import yfinance as yf

import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT"]
start_date = "2020-01-01"
end_date = "2023-01-01"
initial_capital = 1000.0
'''
df = Load_Data(ticker, start=start_date, end=end_date)
df = Generate_Random_Signals(df)
df = Backtest(df, initial_capital=initial_capital)
df = Analyse_Performance(df)

plt.figure(figsize=(12, 6))
plt.plot(df['portfolio_value'], label='Portfolio Value', color='blue')
plt.title(f"Portfolio Value Over Time ({ticker})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
'''
## V2 

def run_backtest_analysis(tickers, start_date, end_date, initial_capital=1000, benchmark_ticker="SPY", plot=True):
    start = time.time()

    # Load portfolio data once
    df = Load_Data_V2(tickers, start=start_date, end=end_date)
    signals = Generate_Random_Signals_V2(df)
    portfolio = Backtest_V2(df, signals, initial_capital=initial_capital)

    # Download benchmark once
    benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date)['Close'].fillna(method='ffill')

    # Compare and plot if needed
    if plot:
        benchmark_returns = benchmark.pct_change().fillna(0)
        benchmark_cumulative = (1 + benchmark_returns).cumprod() * initial_capital
        portfolio_total = portfolio.sum(axis=1)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(portfolio_total, label='Strategy Portfolio')
        plt.plot(benchmark_cumulative, label=benchmark_ticker + ' Benchmark', linestyle='--')
        plt.title('Strategy vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Metrics
    total_portfolio_value = portfolio.sum(axis=1)
    metrics = Calculate_Performance_Metrics(total_portfolio_value)
    for k, v in metrics.items():
        print(f"{k}: {v:.2%}")

    # Calculate alpha and beta
    strategy_returns = portfolio.pct_change().mean(axis=1).dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    alpha, beta = calculate_alpha_beta(strategy_returns, benchmark_returns)

    end = time.time()
    print(f"Calculation took {end - start:.4f} seconds")
    print(f"Alpha: {alpha:.6f}, Beta: {beta:.6f}")

    return {
        "portfolio": portfolio,
        "benchmark": benchmark,
        "metrics": metrics,
        "alpha": alpha,
        "beta": beta,
        "time_elapsed": end - start
    }

results = run_backtest_analysis(tickers, start_date, end_date, initial_capital=initial_capital)