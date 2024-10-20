import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os

def train_and_save_model(data_path: str, model_output_path: str):
    """Train a sentiment analysis model using provided data and save it."""

    # Load dataset
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()

    # Convert sentiment labels to numerical values
    # Note: This model uses 'LABEL_0' for 'notr', 'LABEL_1' for 'positive', 'LABEL_2' for 'negative'
    label_mapping = {'olumlu': 1, 'notr': 0, 'olumsuz': 2}
    labels = [label_mapping[label] for label in labels]

    # Select first 50 and last 50 samples for training (for demo purposes)
    train_texts = texts[:50] + texts[-50:]
    train_labels = labels[:50] + labels[-50:]
    
    # Select all samples
    # train_texts = texts
    # train_labels = labels

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("saribasmetehan/bert-base-turkish-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("saribasmetehan/bert-base-turkish-sentiment-analysis", num_labels=3)

    # Tokenize data
    def tokenize_function(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)

    train_encodings = tokenize_function(train_texts)

    # Create dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].contiguous() for key, val in self.encodings.items()}  # .contiguous() to ensure tensor contiguity
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, train_labels)

    # Define compute_metrics function
    def compute_metrics(p):
        preds = p.predictions.argmax(axis=1)
        acc = accuracy_score(p.label_ids, preds)
        return {"accuracy": acc}

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_path,          # Path to save the model
        num_train_epochs=4,                    # Number of epochs
        per_device_train_batch_size=16,        # Batch size for training
        learning_rate=5e-5,                    # Learning rate used during training
        logging_dir='./logs',                  # Path to logs
        logging_steps=10,                      # Log every 10 steps
        save_steps=500,                        # Save model every 500 steps
        save_total_limit=1,                    # Keep only the last model
        seed=42,                               # Random seed for reproducibility
        evaluation_strategy="steps",           # Evaluate during training at each logging step
        eval_steps=500,                        # Evaluation step
        load_best_model_at_end=True,           # Load the best model at the end of training
        metric_for_best_model="accuracy",      # Choose the best model based on accuracy
    )

    # Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics  # Pass the compute_metrics function to Trainer
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"Model saved to {model_output_path}")

def test_model(model_path: str, text: str):
    """Load a saved model and test with a given input text, showing prediction probabilities."""

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_probs = F.softmax(logits, dim=1)[0]  # softmax işlemi torch ile yapılır
        predicted_class = torch.argmax(predicted_probs).item()

    # Map the predicted class to the sentiment label
    label_mapping = {1: 'olumlu', 0: 'notr', 2: 'olumsuz'}
    sentiment = label_mapping[predicted_class]
    probability = predicted_probs[predicted_class].item()

    print(f"Input: {text}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {probability:.2f})")

if __name__ == "__main__":
    # Define paths
    data_path = 'data/processed/cleaned_df.csv'  # Path to your dataset
    model_output_path = 'data/models/sentiment/saribasmetehan_sentiment_model'  # Path to save the trained model
    
    # Train and save the model
    train_and_save_model(data_path, model_output_path)
    
    # Test the model
    test_input = input("Bir metin girin: ")  # Enter a text to classify
    test_model(model_output_path, test_input)
