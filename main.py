from scraping import Earnings_Call_Scraper
from parser import parse_transcript_to_df
from sentiment import analyze_sentiment
from plotting import SentimentPlotter
import pandas as pd
import os

def ECSA(url, company, quarter, year):
    Earnings_Call_Scraper(url, company, quarter, year)

    txt_path = f"{company}_{quarter}_{year}_transcript.txt"
    df = parse_transcript_to_df(txt_path)

    csv_path = f"{company}_{quarter}_{year}_by_speaker.csv"
    df.to_csv(csv_path, index=False)

    df[['sentiment', 'confidence']] = df['text'].apply(analyze_sentiment)
    df.to_csv(csv_path.replace(".csv", "_finbert.csv"), index=False)

    plotter = SentimentPlotter(df, company, quarter, year)
    plotter.plot_by_speaker()
    plotter.plot_distribution()

if __name__ == "__main__":
    ECSA(
    "https://www.fool.com/earnings/call-transcripts/2024/04/25/microsoft-msft-q3-2024-earnings-call-transcript/",
    "Microsoft", "Q3", "2024"
)