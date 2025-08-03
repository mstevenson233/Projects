import pandas as pd
import os

def Comparative_ECSA(earning_calls):
    comparisons = []

    for call in earning_calls:
        print(f"Processing {call['company']} {call['quarter']} {call['year']} ...")
        ECSA(call["url"], call["company"], call["quarter"], call["year"])

        # Adjust path for your CSV files, assuming you save in 'data/csv/'
        csv_path = f"data/csv/{call['company']}_{call['quarter']}_{call['year']}_finbert.csv"
        df = pd.read_csv(csv_path)

        total = len(df)
        value_counts = df['sentiment'].value_counts()
        results = {
            "company": call["company"],
            "quarter": call["quarter"],
            "year": call["year"],
            "positive_%": round(value_counts.get("positive", 0) / total * 100, 2),
            "neutral_%": round(value_counts.get("neutral", 0) / total * 100, 2),
            "negative_%": round(value_counts.get("negative", 0) / total * 100, 2),
            "overall_counts": total,
            "avg_positive_conf": round(df[df['sentiment'] == 'positive']['confidence'].mean(), 2),
            "avg_neutral_conf": round(df[df['sentiment'] == 'neutral']['confidence'].mean(), 2),
            "avg_negative_conf": round(df[df['sentiment'] == 'negative']['confidence'].mean(), 2)
        }
        comparisons.append(results)

    comparative_df = pd.DataFrame(comparisons)

    # Create a folder to save summary if you want
    os.makedirs("data/summary", exist_ok=True)
    summary_path = "data/summary/comparative_sentiment_summary.csv"
    comparative_df.to_csv(summary_path, index=False)
    print(f"Comparative summary saved at {summary_path}")

    return comparative_df