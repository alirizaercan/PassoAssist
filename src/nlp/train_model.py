# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import numpy as np

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.fillna('', inplace=True)
    df['aksiyon'] = df['severity'].apply(lambda x: 1 if x in [1, 2] else 0)
    for column in ['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig_kart', 'diger']:
        df[column] = df.apply(lambda row: 1 if row['metin'].lower() in row[column].lower() else 0, axis=1)
    return df

def preprocess_data(df, column_name):
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    def tokenize_function(examples):
        return tokenizer(examples['metin'], padding="max_length", truncation=True)
    dataset = Dataset.from_pandas(df[['metin', column_name]])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def compute_metrics(pred):
    metric = load_metric("accuracy")
    predictions = np.argmax(pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=pred.label_ids)

def train_model(column_name, num_labels):
    df = load_and_preprocess_data('data/raw/merged_df.csv')
    tokenized_dataset = preprocess_data(df, column_name)
    train_dataset, eval_dataset = train_test_split(tokenized_dataset, test_size=0.2)
    model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=f'./results/{column_name}',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(f'data/models/{column_name}_model')

if __name__ == "__main__":
    for column_name in ['severity', 'bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig_kart', 'diger']:
        train_model(column_name, 2)
