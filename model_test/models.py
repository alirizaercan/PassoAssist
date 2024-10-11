import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.multiclass import OneVsRestClassifier

def train_random_forest(data_path, model_output_path):
    # Load the data
    df = pd.read_csv(data_path)

    # Features (X) and target labels (y)
    X = df['text']
    y = df[['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig kart', 'diger']]

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train model
    model = OneVsRestClassifier(RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ))

    model.fit(X_train, y_train)

    # Save model
    joblib.dump((model, tfidf), model_output_path)

    # Evaluate performance
    y_pred = model.predict(X_test)
    print("\nRandom Forest Model Performance:")
    print(classification_report(y_test, y_pred, target_names=y.columns))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

def train_gradient_boosting(data_path, model_output_path):
    df = pd.read_csv(data_path)

    X = df['text']
    y = df['severity']

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump((model, tfidf), model_output_path)

    y_pred = model.predict(X_test)
    print("\nGradient Boosting Model Performance:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

def train_transformer_model(data_path, model_output_path):
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()

    label_mapping = {'olumlu': 1, 'notr': 0, 'olumsuz': 2}
    labels = [label_mapping[label] for label in labels]

    tokenizer = AutoTokenizer.from_pretrained("saribasmetehan/bert-base-turkish-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("saribasmetehan/bert-base-turkish-sentiment-analysis", num_labels=3)

    train_encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].contiguous() for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, labels)

    training_args = TrainingArguments(
        output_dir=model_output_path,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)

def test_model(model_output_path, input_text):
    # Load model and tokenizer
    if "transformer" in model_output_path:
        model = AutoModelForSequenceClassification.from_pretrained(model_output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_output_path)

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1).item()
        print(f"Tahmin Edilen Duygu: {predictions}")
    else:
        model, tfidf = joblib.load(model_output_path)
        input_tfidf = tfidf.transform([input_text])
        prediction = model.predict(input_tfidf)

        # Show predictions
        print("Tahmin Edilen Etiketler:")
        print(prediction)

if __name__ == "__main__":
    data_path = 'data/processed/cleaned_df.csv'

    # Train Random Forest model
    rf_model_output_path = 'data/models/multilabel/multilabelclassifier.pkl'
    train_random_forest(data_path, rf_model_output_path)

    # Train Gradient Boosting model
    gb_model_output_path = 'data/models/severity_classifier.pkl'
    train_gradient_boosting(data_path, gb_model_output_path)

    # Train Transformer model
    transformer_model_output_path = 'data/models/sentiment_classifier'
    train_transformer_model(data_path, transformer_model_output_path)

    # Test models with user input
    while True:
        test_input = input("Bir metin girin (çıkmak için 'q' yazın): ")
        if test_input.lower() == 'q':
            break
        print("\nRandom Forest Model Tahminleri:")
        test_model(rf_model_output_path, test_input)
        
        print("\nGradient Boosting Model Tahminleri:")
        test_model(gb_model_output_path, test_input)
        
        print("\nTransformer Model Tahminleri:")
        test_model(transformer_model_output_path, test_input)
