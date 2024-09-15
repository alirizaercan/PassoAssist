import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os

def train_and_save_model(data_path: str, model_output_path: str):
    """ Train a 3-class sentiment analysis model and save it. """
    
    # Veri yükleme
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()  # Metin sütunu
    labels = df['sentiment'].tolist()  # Sentiment sütunu (olumlu, olumsuz, nötr)

    # Sentiment etiketlerini sayısal değerlere dönüştürme
    label_mapping = {'olumlu': 2, 'notr': 1, 'olumsuz': 0}  # Olumlu: 2, Nötr: 1, Olumsuz: 0
    labels = [label_mapping[label] for label in labels]

    # Eğitim ve test verisi ayırma
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenizer ve model yükleme
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=3)  # Üç sınıf (olumlu, nötr, olumsuz)

    # Veriyi tokenize etme
    def tokenize_function(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # Eğitim ve test verisini token haline getirme
    train_encodings = tokenize_function(X_train)
    test_encodings = tokenize_function(X_test)
    
    # Tensor haline getirme
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, y_train)
    test_dataset = Dataset(test_encodings, y_test)

    # Model eğitim ayarları
    training_args = TrainingArguments(
        output_dir=model_output_path,          # Çıktı yolu
        num_train_epochs=3,                    # Eğitim dönemi sayısı
        per_device_train_batch_size=8,         # Eğitim batch boyutu
        per_device_eval_batch_size=8,          # Değerlendirme batch boyutu
        evaluation_strategy="steps",           # Her birkaç adımda bir değerlendirme
        save_steps=1000,                       # Model kaydetme adımı
        save_total_limit=2,                    # Kaç model saklanacak
        logging_dir='./logs',                  # Log dosyaları için dizin
        logging_steps=10,                      # Loglama adım sayısı
        load_best_model_at_end=True,           # En iyi modeli eğitim sonunda yükle
    )

    # Trainer nesnesi
    trainer = Trainer(
        model=model,                           # Model
        args=training_args,                    # Eğitim argümanları
        train_dataset=train_dataset,           # Eğitim veri seti
        eval_dataset=test_dataset              # Test veri seti
    )

    # Modeli eğit
    trainer.train()

    # Modeli kaydet
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # Model ve veri yollarını tanımlama
    data_path = 'data/processed/cleaned_df.csv'  # Verinizin yolu
    model_output_path = 'data/models/bert_sentiment_model'  # Kaydedilecek modelin yolu
    train_and_save_model(data_path, model_output_path)
