# train_model.py
import re
import spacy
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ENTITY RECOGNITION MODEL
def train_and_save_entity_model(data_path, model_output_path):
    nlp = spacy.load("tr_core_news_trf")
    entities = ["passo", "passolig", "passolig kart"]
    corrected_spellings = {
        "passolg": "passolig",
        "passolig krt": "passolig kart",
    }

    def correct_spelling(sentence):
        words = sentence.split()
        corrected_words = [corrected_spellings.get(word.lower(), word) for word in words]
        return ' '.join(corrected_words)

    def find_entities(sentence, entity_list):
        corrected_sentence = correct_spelling(sentence)
        found_entities = []
        for entity in entity_list:
            pattern = rf'\b{entity}(?:\w+)?\b'
            if re.search(pattern, corrected_sentence, re.IGNORECASE):
                found_entities.append(entity)
        return found_entities

    model = {
        "entities": entities,
        "spelling_correction": corrected_spellings
    }

    joblib.dump(model, model_output_path)
    print(f"Entity Recognition Model saved to: {model_output_path}")

# TOPIC CLASSIFICATION MODEL
def train_and_save_konu_model(data_path, model_output_path):
    data = pd.read_csv(data_path)
    X = data['text']
    y = pd.Categorical(data['konu'], categories=[
        'cagri merkezi yetkinlik', 'diger', 'genel', 'odeme', 'uygulama',
        'iptal', 'degisiklik', 'uyelik', 'iade', 'transfer', 'fatura'
    ]).codes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Topic Model Accuracy: {accuracy:.2f}")
    joblib.dump(model, model_output_path)
    print(f"Topic Classification Model saved to: {model_output_path}")

# MULTILABEL CLASSIFICATION MODEL
def train_and_save_multilabel_model(data_path, model_output_path):
    df = pd.read_csv(data_path)
    X = df['text']
    y = df[['bilet', 'musteri_hizmetleri', 'odeme', 'uygulama', 'passolig', 'passolig kart', 'diger']]

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    model.fit(X_train, y_train)
    
    joblib.dump((model, tfidf), model_output_path)
    print(f"Multilabel Classification Model saved to: {model_output_path}")

# SENTIMENT ANALYSIS MODEL
def train_and_save_sentiment_model(data_path, model_output_path):
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = [ {'olumlu': 1, 'notr': 0, 'olumsuz': 2}[label] for label in df['sentiment'].tolist() ]
    
    tokenizer = AutoTokenizer.from_pretrained("saribasmetehan/bert-base-turkish-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("saribasmetehan/bert-base-turkish-sentiment-analysis", num_labels=3)
    
    train_encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    
    training_args = TrainingArguments(
        output_dir=model_output_path,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        seed=42
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=train_encodings, train_labels=labels)
    trainer.train()
    
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"Sentiment Analysis Model saved to: {model_output_path}")

# SEVERITY CLASSIFICATION MODEL
def train_and_save_severity_model(data_path, model_output_path):
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['severity']
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump((model, tfidf), model_output_path)
    print(f"Severity Classification Model saved to: {model_output_path}")

if __name__ == "__main__":
    data_path = 'data/processed/cleaned_df.csv'

    # Model output paths
    entity_model_path = 'data/models/entity_model.joblib'
    konu_model_path = 'data/models/konu_model.joblib'   
    multilabel_model_path = 'data/models/multilabel/multilabelclassifier.joblib'
    sentiment_model_path = 'data/models/sentiment/saribasmetehan_sentiment_model'
    severity_model_path = 'data/models/severity_classifier.joblib'
    

    
    # Train and save models
    train_and_save_entity_model(data_path, entity_model_path)
    train_and_save_konu_model(data_path, konu_model_path)
    train_and_save_multilabel_model(data_path, multilabel_model_path)
    train_and_save_sentiment_model(data_path, sentiment_model_path)
    train_and_save_severity_model(data_path, severity_model_path)
