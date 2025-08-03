from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd

model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
finbert_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    result = finbert_sentiment(text[:512])[0]
    return pd.Series([result['label'], result['score']])