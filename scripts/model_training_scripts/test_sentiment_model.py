import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def load_model(model_path: str):
    """Load the model and tokenizer from the specified path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_sentiment(model, tokenizer, text: str):
    """Predict the sentiment of the input text using the given model and tokenizer."""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_probs = F.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(predicted_probs).item()

    # Map the predicted class to the sentiment label
    label_mapping = {1: 'olumlu', 0: 'notr', 2: 'olumsuz'}
    sentiment = label_mapping[predicted_class]
    probability = predicted_probs[predicted_class].item()

    return sentiment, probability

def main():
    model_path = 'data/models/sentiment/saribasmetehan_sentiment_model'  # Path to the saved model
    tokenizer, model = load_model(model_path)
    
    print("Sentiment Analysis Model is ready for testing. Type 'exit' to stop the program.")

    while True:
        text = input("Bir metin girin: ")
        if text.lower() == 'exit':
            print("Programdan çıkılıyor.")
            break

        sentiment, confidence = predict_sentiment(model, tokenizer, text)
        print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
