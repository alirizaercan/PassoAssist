import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os

def train_and_save_model(data_path: str, model_output_path: str):
    """ Train a 3-class sentiment analysis model and save it. """
    
    # Load dataset
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()  # Text column
    labels = df['sentiment'].tolist()  # Sentiment column (positive, neutral, negative)

    # Convert sentiment labels to numerical values
    label_mapping = {'olumlu': 2, 'notr': 1, 'olumsuz': 0}
    labels = [label_mapping[label] for label in labels]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=3)

    # Tokenize data
    def tokenize_function(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

    train_encodings = tokenize_function(X_train)
    test_encodings = tokenize_function(X_test)

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

    train_dataset = Dataset(train_encodings, y_train)
    test_dataset = Dataset(test_encodings, y_test)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_path,          # Path to save the model
        num_train_epochs=3,                    # Number of epochs
        per_device_train_batch_size=8,         # Batch size for training
        per_device_eval_batch_size=8,          # Batch size for evaluation
        evaluation_strategy="steps",           # Evaluate every few steps
        save_steps=1000,                       # Save model every 1000 steps
        save_total_limit=2,                    # Keep only 2 models
        logging_dir='./logs',                  # Path to logs
        logging_steps=10,                      # Log every 10 steps
        load_best_model_at_end=True,           # Load the best model at the end of training
    )

    # Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # Define paths
    data_path = 'data/processed/cleaned_df.csv'  # Path to your dataset
    model_output_path = 'data/models/bert_sentiment_model'  # Path to save the trained model
    train_and_save_model(data_path, model_output_path)
