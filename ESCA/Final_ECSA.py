import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


def ECSA(url, company, quarter, year):
    url = url
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/90.0.4430.93 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Get all paragraph elements
    paragraphs = soup.find_all("p")

    # Find where the actual transcript starts
    start_idx = None
    for i, p in enumerate(paragraphs):
        text = p.get_text(strip=True)
        if text.startswith("Operator") or "Prepared Remarks" in text:
            start_idx = i
            break

    # Extract from start index until "Duration:" is hit
    transcript_lines = []
    if start_idx is not None:
        for p in paragraphs[start_idx:]:
            text = p.get_text(strip=True)
            if text.startswith("Duration:"):
                break
            transcript_lines.append(text)

        transcript_text = "\n\n".join(transcript_lines)

        with open(f"{company}_{quarter}_{year}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript_text)

        print("✅ Transcript saved successfully and ended at 'Duration'.")
    else:
        print("❌ Could not find the start of the transcript.")

    
    
    ## Prepare Call for Analysis ##
    ###############################
    # Raw transcript
    with open(f"{company}_{quarter}_{year}_transcript.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Normalize whitespace and remove non-informative lines
    text = re.sub(r"\n+", "\n", raw_text)  # Remove multiple newlines

    # Formatted to detect speaker -- title and Operator
    pattern = r"(?P<speaker>[A-Z][a-z]+(?: [A-Z][a-z]+)*|Operator)\s*--\s*.*?\n(?P<text>.*?)(?=(?:[A-Z][a-z]+(?: [A-Z][a-z]+)*|Operator)\s*--\s*|Duration:|$)"


    matches = re.finditer(pattern, text, flags=re.DOTALL)

    speaker_data = []

    # Filter and collect data
    for match in matches:
        speaker = match.group("speaker").strip()
        speech = match.group("text").strip()
        speaker_data.append({"speaker": speaker, "text": speech})
        
    df = pd.DataFrame(speaker_data)
    df.to_csv(f"{company}_{quarter}_{year}_transcript_by_speaker.csv", index=False)
    print(len(df))
    ###############################
    ## Analyse Call With FinBert ##
    ###############################

    # Load FinBERT model and tokenizer
    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Create sentiment analysis pipeline
    finbert_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    df = pd.read_csv(f"{company}_{quarter}_{year}_transcript_by_speaker.csv")

    # Limit to first 512 tokens (BERT limit)
    def analyze_sentiment(text):
        result = finbert_sentiment(text[:512])[0]
        return pd.Series([result['label'], result['score']])

    df[['sentiment', 'confidence']] = df['text'].apply(analyze_sentiment)

    # Save results
    df.to_csv(f"{company}_{quarter}_{year}_finbert_sentiment.csv", index=False)

    # Overview of results

    #print(df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative']))
    #print("Positive avg confidence:", df[df['sentiment'] == 'positive']['confidence'].mean())
    #print("Neutral avg confidence:", df[df['sentiment'] == 'neutral']['confidence'].mean()) 
    #print("Negative avg confidence:", df[df['sentiment'] == 'negative']['confidence'].mean())


    ###############################
    ## Visualisation of Analysis ##
    ###############################

    # sentiment by speaker
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="speaker", hue="sentiment", order=df['speaker'].value_counts().index)
    plt.title("Sentiment by Speaker")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.xlabel("Speaker")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(f"{company}_{quarter}_{year}_sentiment_by_speaker.png")
    plt.close()


    # Overall sentiment count
    sns.countplot(data=df, x="sentiment", order=["positive", "neutral", "negative"])
    plt.title("Sentiment Distribution (FinBERT)")
    plt.ylabel("Number of Responses")
    plt.xlabel("Sentiment")
    plt.savefig(f"{company}_{quarter}_{year}_sentiment_distribution.png")
    plt.close()


# Multi_ECSA
def Comparative_ECSA(earning_calls):

    comparisons=[]
    for call in earning_calls:
        ECSA(call["url"], call["company"], call["quarter"], call["year"])
        
        df = pd.read_csv(f"{call['company']}_{call['quarter']}_{call['year']}_finbert_sentiment.csv")

        total = len(df)
        value_counts = df['sentiment'].value_counts()
        results={
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

    
    

    return pd.DataFrame(comparisons).to_csv("comparative_sentiment_summary.csv", index=False)


# Call # 
earning_calls = [
    {
        "url": "https://www.fool.com/earnings/call-transcripts/2024/07/23/alphabet-googl-q2-2024-earnings-call-transcript/",
        "company": "Google",
        "quarter": "Q2",
        "year": "2024"
    },
    {
        "url": "https://www.fool.com/earnings/call-transcripts/2024/04/25/alphabet-googl-q1-2024-earnings-call-transcript/",
        "company": "Google",
        "quarter": "Q1",
        "year": "2024"
    },
    {   "url": "https://www.fool.com/earnings/call-transcripts/2024/01/30/alphabet-googl-q4-2023-earnings-call-transcript/",
        "company": "Google",
        "quarter": "Q4",
        "year": "2023"
    },
    {   "url": "https://www.fool.com/earnings/call-transcripts/2023/07/25/alphabet-googl-q2-2023-earnings-call-transcript/",
        "company": "Google",
        "quarter": "Q2",
        "year": "2023"
    }
    ]
summary_df = Comparative_ECSA(earning_calls)
print(summary_df)
