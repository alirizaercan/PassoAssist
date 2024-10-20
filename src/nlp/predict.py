# predict.py
# Bu dosya eğitilmiş model ile tahmin yapma işlemlerini içerir.
# Yeni veriler üzerinde model tahminleri gerçekleştirir.

# predict.py
# predict.py
import pandas as pd
import joblib
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# CSV'yi okuma
df = pd.read_csv('data/processed/cleaned_df.csv')

# 'konu' labels
konu_labels = [
    'cagri merkezi yetkinlik', 'diger', 'genel', 'odeme', 'uygulama',
    'iptal', 'degisiklik', 'uyelik', 'iade', 'transfer', 'fatura'
]

# Model paths
entity_model_path = 'data/models/entity_model.joblib'
konu_model_path = 'data/models/konu_model.joblib'
sentiment_model_path = 'data/models/sentiment/saribasmetehan_sentiment_model'
severity_model_path = 'data/models/severity_classifier.joblib'
multilabel_model_path = 'data/models/multilabel/multilabelclassifier.joblib'

# Load models
def load_model(path):
    return joblib.load(path)

try:
    entity_model = load_model(entity_model_path)
except Exception as e:
    print(f"Error loading entity model: {e}")

try:
    konu_model = load_model(konu_model_path)
except Exception as e:
    print(f"Error loading konu model: {e}")

try:
    loaded_model = load_model(severity_model_path)
    if isinstance(loaded_model, tuple):
        severity_model = loaded_model[0]
    else:
        severity_model = loaded_model
except Exception as e:
    print(f"Error loading severity model: {e}")

try:
    multilabel_model, tfidf_vectorizer = load_model(multilabel_model_path)
except Exception as e:
    print(f"Error loading multilabel model: {e}")

try:
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
except Exception as e:
    print(f"Error loading sentiment model: {e}")

# Entity prediction helper functions
def correct_spelling(sentence):
    corrected_spellings = {
        "passolg": "passolig",
        "passolig krt": "passolig kart",
    }
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

def process_sentence(sentence, entity_list):
    found_entities = find_entities(sentence, entity_list)
    if found_entities:
        return "; ".join(found_entities)
    return "No entity found."

# Prediction functions
def predict_entity(input_text):
    entities = ["passo", "passolig", "passolig kart"]
    return process_sentence(input_text, entities)

def predict_konu(input_text):
    predicted_label = konu_model.predict([input_text])
    return konu_labels[predicted_label[0]]

def predict_sentiment(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        sentiment_logits = sentiment_model(**inputs).logits
        sentiment_prediction = sentiment_logits.argmax().item()
        predicted_probs = torch.softmax(sentiment_logits, dim=1)[0]

    label_mapping = {1: 'olumlu', 0: 'notr', 2: 'olumsuz'}
    sentiment_label = label_mapping[sentiment_prediction]
    sentiment_confidence = predicted_probs[sentiment_prediction].item()

    return sentiment_label, sentiment_confidence

def predict_severity(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    severity_prediction = severity_model.predict(input_tfidf)[0]

    if severity_prediction == 2:
        action_status = 1
        action_message = "Acil Harekete Geçin!"
    elif severity_prediction == 1:
        action_status = 1
        action_message = "Aksiyon Almanız Önerilir."
    else:
        action_status = 0
        action_message = "Harekete Geçmeye Gerek Yok."
    
    return severity_prediction, action_status, action_message

def predict_multilabel(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    multilabel_prediction = multilabel_model.predict(input_tfidf)

    return {
        'bilet': multilabel_prediction[0][0],
        'musteri_hizmetleri': multilabel_prediction[0][1],
        'odeme': multilabel_prediction[0][2],
        'uygulama': multilabel_prediction[0][3],
        'passolig': multilabel_prediction[0][4],
        'passolig kart': multilabel_prediction[0][5],
        'diger': multilabel_prediction[0][6],
    }

# Prediction functions
def predict_all_models(input_text):
    # Bu kısmı önceki koddan aldık
    entity_prediction = predict_entity(input_text)
    konu_prediction = predict_konu(input_text)
    sentiment_prediction, sentiment_confidence = predict_sentiment(input_text)
    severity_prediction, action_status, action_message = predict_severity(input_text)
    multilabel_prediction = predict_multilabel(input_text)

    return {
        'Entity': entity_prediction,
        'Konu': konu_prediction,
        'Sentiment': {
            'label': sentiment_prediction,
            'confidence': sentiment_confidence
        },
        'Severity': {
            'severity_label': severity_prediction,
            'action_status': action_status,
            'action_message': action_message
        },
        'Multilabel': multilabel_prediction
    }

# Tahmin ve veri saklama
predictions = []
for index, row in df.iterrows():
    user_input = row['text']
    model_outputs = predict_all_models(user_input)
    
    # Sonuçları bir dict olarak kaydet
    data = {
        'text': user_input,
        'entity': model_outputs['Entity'] if 'Entity' in model_outputs else '',
        'sentiment': model_outputs['Sentiment']['label'] if 'Sentiment' in model_outputs else '',
        'konu': model_outputs['Konu'] if 'Konu' in model_outputs else '',
        'severity': model_outputs['Severity']['severity_label'] if 'Severity' in model_outputs else '',
        'bilet': model_outputs['Multilabel']['bilet'] if 'Multilabel' in model_outputs else '',
        'musteri_hizmetleri': model_outputs['Multilabel']['musteri_hizmetleri'] if 'Multilabel' in model_outputs else '',
        'odeme': model_outputs['Multilabel']['odeme'] if 'Multilabel' in model_outputs else '',
        'uygulama': model_outputs['Multilabel']['uygulama'] if 'Multilabel' in model_outputs else '',
        'passolig': model_outputs['Multilabel']['passolig'] if 'Multilabel' in model_outputs else '',
        'passolig kart': model_outputs['Multilabel']['passolig kart'] if 'Multilabel' in model_outputs else '',
        'diger': model_outputs['Multilabel']['diger'] if 'Multilabel' in model_outputs else '',
        'aksiyon': model_outputs['Severity']['action_status'] if 'Severity' in model_outputs else '',
        'sentiment_confidence': model_outputs['Sentiment']['confidence'] if 'Sentiment' in model_outputs else '',
    }
    
    predictions.append(data)

# Tahmin edilen verileri DataFrame'e çevirme
predictions_df = pd.DataFrame(predictions)

# DataFrame'i CSV'ye kaydet
predictions_df.to_csv('data/processed/predictions.csv', index=False)