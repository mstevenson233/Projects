import matplotlib as plt
import pandas as pd
import numpy as np
import yfinance as yf

 #df = yf.download("AAPL", start="2022-01-01", end="2023-01-01") # download data

#df = df.sort_index() # clean data
#df = df.dropna()

#df.to_csv("data/aapl.csv") # save data

def Load_Data(ticker, start, end):
    df = yf.download(ticker, start, end)
    df = df.sort_index() # clean data
    df = df.dropna()
    return df


def Load_Data_V2(tickers: list[str], start, end):
    df = yf.download(tickers, start, end)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']  # now selects only the adjusted close prices per ticker
    else:
        df = df[['Close']]  # keep as DataFrame
        df.columns = [tickers[0]]

    df.columns.name = None  # remove name from column index
    df.columns = [str(col) for col in df.columns]
    df = df.sort_index() # clean data
    df = df.dropna()
    return df
