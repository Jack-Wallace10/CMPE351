import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the model name (a BERT-based model fine-tuned on SST-2 for sentiment analysis)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define a function that performs sentiment analysis on a given text
def analyze_sentiment(text):
    # Return None if the text is empty or not a valid string
    if not isinstance(text, str) or not text.strip():
        return None

    # Tokenize the text and limit the length to 512 tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Get the model outputs (logits)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    
    # Get the probability of the positive class (index 1)
    positive_prob = float(probabilities[0][1])
    
    # Convert to a scale from -1 to +1
    # For this model, label 0 corresponds to NEGATIVE and 1 corresponds to POSITIVE sentiment
    # We'll use the probability of the positive class to create a score from -1 to +1
    sentiment_score = (positive_prob * 2) - 1
    
    return sentiment_score

# Load your CSV file that contains columns such as: date, stock, title, article
try:
    df = pd.read_csv("stocks_pre_sentiment.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("stocks_pre_sentiment.csv", encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv("stocks_pre_sentiment.csv", encoding='cp1252')

# Apply sentiment analysis to both the "title" and "article" columns
df["sentiment_title"] = df["title"].apply(analyze_sentiment)
df["sentiment_article"] = df["article"].apply(analyze_sentiment)

# Save the resulting DataFrame (with the new sentiment columns) to a new CSV file
output_filename = "stocks_with_sentiment_custom.csv"
df.to_csv(output_filename, index=False)

print(f"Sentiment analysis completed. Results saved as '{output_filename}'.")
