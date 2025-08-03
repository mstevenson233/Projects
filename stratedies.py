import matplotlib as plt
import pandas as pd
import numpy as np
import yfinance as yf


 ## Random 50/50
def Generate_Random_Signals(df):
    df=df.copy()
    signals = np.random.choice([-1, 1], size=len(df))
    df["Signal"]=signals
    return df

def Generate_Random_Signals_V2(df):
    signals = pd.DataFrame(index=df.index, columns=df.columns)
    for ticker in df.columns:
        signals[ticker] = np.random.choice([-1, 1], size=len(df))
    return signals
