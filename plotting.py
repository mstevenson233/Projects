import matplotlib.pyplot as plt
import seaborn as sns
import os

class SentimentPlotter:
    def __init__(self, df, company, quarter, year):
        self.df = df
        self.company = company
        self.quarter = quarter
        self.year = year

    def plot_by_speaker(self):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x="speaker", hue="sentiment", order=self.df['speaker'].value_counts().index)
        plt.title("Sentiment by Speaker")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.xlabel("Speaker")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        filename = f"{self.company}_{self.quarter}_{self.year}_by_speaker.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

    def plot_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x="sentiment", order=["positive", "neutral", "negative"])
        plt.title("Sentiment Distribution (FinBERT)")
        plt.ylabel("Number of Responses")
        plt.xlabel("Sentiment")
        filename = f"{self.company}_{self.quarter}_{self.year}_distribution.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")