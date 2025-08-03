import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparative_summary(df):
    # Melt the DataFrame for easier plotting
    df_melt = df.melt(
        id_vars=["company", "quarter", "year"],
        value_vars=["positive_%", "neutral_%", "negative_%"],
        var_name="sentiment",
        value_name="percentage"
    )
    
    # Create a unique label for each earnings call
    df_melt["label"] = df["company"] + " " + df["quarter"] + " " + df["year"]

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x="label", y="percentage", hue="sentiment")
    plt.title("Sentiment Comparison Across Earnings Calls")
    plt.ylabel("Sentiment Percentage")
    plt.xlabel("Earnings Call")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    os.makedirs("comparisons", exist_ok=True)
    plt.savefig("comparisons/comparative_sentiment_plot.png")
    plt.close()
    print("Saved plot: comparisons/comparative_sentiment_plot.png")